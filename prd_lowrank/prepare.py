import os, time, json
from pathlib import Path
import numpy as np, torch
from prd_physical.common import N, Preprocess, write
from prd_physical.prepare import crop_meta
from prd_physical.physics import Simulator
from prd_strategy.augment import prepare_observation
from .settings import *
from .common import status, check_sources, ram_guard, load_backbone, inferred
from .data import FIELDS, packet

def maps_for(path, count):
    path.mkdir(parents=True, exist_ok=True)
    maps = {}
    for k, shape in FIELDS.items():
        p = path / (k + '.npy')
        if p.exists():
            a = np.load(p, mmap_mode='r+')
            assert a.shape == (count,) + shape
        else:
            a = np.lib.format.open_memmap(p, mode='w+', dtype='f4', shape=(count,) + shape)
        maps[k] = a
    return maps

def main(destination=OUT, count=TRAIN_N):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'prepared.json').exists():
        assert json.loads((destination / 'prepared.json').read_text())['training_sources'] == count
        return
    torch.set_num_threads(2)
    check_sources()
    ram_guard(destination)
    cache = destination / 'cache'
    basis = np.load(PROBE / 'training_direction_basis.npz')['vectors'][:, :RANK]
    model = load_backbone()
    pre = Preprocess(np.load(CAL)['psd'], 'white')
    truth = np.load(CACHE / 'train_truth.npy', mmap_mode='r')
    mask = np.load(CACHE / 'train_mask.npy', mmap_mode='r')
    params = np.load(CACHE / 'train_parameters.npy', mmap_mode='r')
    seeds = np.load(CACHE / 'train_seeds.npy', mmap_mode='r')
    old = {k: np.load(CACHE / ('train_' + k + '.npy'), mmap_mode='r') for k in ('input', 'q', 'q_low', 'bands')}
    sim = None
    for j in range(VIEWS):
        path = cache / f'view{j}'
        maps = maps_for(path, count)
        progress = path / 'progress.json'
        completed = json.loads(progress.read_text())['completed'] if progress.exists() else 0
        sign = np.ones(count, 'f4') if j == 0 else np.random.default_rng(SEED + 4 + j).choice([-1., 1.], count).astype('f4')
        t, begin, last = time.monotonic(), completed, 0.
        if j and sim is None:
            sim = Simulator()
        for lo in range(completed, count, INFERENCE_BATCH):
            hi = min(lo + INFERENCE_BATCH, count)
            if j == 0:
                u = old['input'][lo:hi]
                v = {k: torch.from_numpy(np.array(a[lo:hi], copy=True)) for k, a in old.items()}
                with torch.inference_mode():
                    p = model(v['input'], v['q'], torch.from_numpy(np.array(mask[lo:hi], copy=True)), v['bands'], v['q_low']).numpy()
            else:
                noise = []
                for i in range(lo, hi):
                    start, expected_mask = crop_meta(params[i], int(seeds[i]))
                    assert np.array_equal(expected_mask, mask[i])
                    n = sim.noise(NOISE_SEED + (j-1)*TRAIN_N + i)[:, start:start+N].astype('f4')
                    noise.append(n)
                u, target = prepare_observation(truth[lo:hi], np.stack(noise), mask[lo:hi], pre, sign[lo:hi])
                assert np.array_equal(target, np.load(CACHE / 'train_target.npy', mmap_mode='r')[lo:hi] * sign[lo:hi, None, None])
                p = inferred(model, u, mask[lo:hi])
            assert np.isfinite(p).all() and np.isfinite(u).all()
            values = packet(u, p, mask[lo:hi], basis, sign[lo:hi])
            for k in FIELDS:
                maps[k][lo:hi] = values[k]
            if time.monotonic() - last > 30 or hi == count:
                for a in maps.values():
                    a.flush()
                elapsed = time.monotonic() - t
                rate = (hi-begin) / max(elapsed, .001)
                write(progress, dict(completed=hi, count=count, view=j))
                status(destination, 'preparing_training_features', view=j, completed=hi, total=count, sources_per_second=rate, current_view_remaining_seconds=(count-hi)/max(rate, .001), elapsed_seconds=elapsed, device='cpu')
                last = time.monotonic()
                ram_guard(destination)
        del maps
    vp = np.load(PROBE / 'frequency_snapshot_validation_prediction.npy')
    vu = np.load(CACHE / 'validation_input.npy', mmap_mode='r')
    vm = np.load(CACHE / 'validation_mask.npy', mmap_mode='r')
    values = packet(vu, vp, vm, basis, np.ones(VAL_N))
    vmaps = maps_for(cache / 'validation', VAL_N)
    for k in FIELDS:
        vmaps[k][:] = values[k]
        vmaps[k].flush()
    write(destination / 'prepared.json', dict(passed=True, training_sources=count, views=VIEWS, training_observations=count*VIEWS, validation_sources=VAL_N, basis_training_sources=TRAIN_N, new_noise_seed_range=[NOISE_SEED, NOISE_SEED+(VIEWS-1)*TRAIN_N-1], noise_source='same physical Simulator.noise; independent Gaussian draws; full-parent process then actual crop', global_sign=True, observation='G*h+n', augmented_features_recomputed=True, random_recrop=False, test_read=False))
    status(destination, 'prepared', count=count, views=VIEWS, test_read=False)

if __name__ == '__main__':
    main()
