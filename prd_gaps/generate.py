"""Create source-disjoint datasets. Every attempted source is retained or aborts.

Example: python -m prd_gaps.generate --output /path/data --train 64 --val 16
Large production generation should follow the acceptance gates in README.md.
"""
import argparse, json, time, hashlib
import importlib.metadata
from pathlib import Path
import numpy as np
from .physics import Simulator, draw_parameters
from .preprocess import design_fir, make_features

def generate(destination, n_train=32, n_val=8, n_test=0, calibration_count=128):
    out = Path(destination); out.mkdir(parents=True, exist_ok=False)
    sim = Simulator()
    psd = sim.calibrate_psd(calibration_count)
    kernel, scale = design_fir(psd, sim.fs, radius=32)
    np.savez(out/'calibration.npz', psd=psd, kernel=kernel, scale=scale)
    manifest = dict(schema=2, observation='G*(s+n)', fs_hz=sim.fs, n=sim.n,
                    approximant='IMRPhenomTHM', noise_model='scirdv1',
                    mass_source_msun=[1e5, 5e6], redshift=[1., 6.], q=[.3, 1.],
                    noise_calibration_records=calibration_count,
                    spectral_floor_fraction=1e-4, fir_radius_samples=32,
                    gap_durations_seconds=[33., 53.], all_source_outcomes_retained=True,
                    source_split_seeds=dict(train=31000000, val=41000000, test=51000000),
                    waveform_peak_time_jitter_seconds=[1024., 3072.],
                    noise_scope='Stationary A/E secondary noise; link-propagated availability',
                    source_code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in Path(__file__).parent.glob('*.py')},
                    phase_reference='f_ref=0.02/(G M_detector/c^3)',
                    source_alignment='subsample polarization-envelope peak; cubic source resampling',
                    splits={}, created_unix=time.time())
    manifest['versions'] = {}
    for name in ['numpy', 'scipy', 'torch', 'pycbc', 'lalsuite', 'fastlisaresponse', 'lisaanalysistools']:
        try: manifest['versions'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: manifest['versions'][name] = 'unavailable'
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for split, count in [('train', n_train), ('val', n_val), ('test', n_test)]:
        (out/split).mkdir(); entries = []
        for j in range(count):
            seed = manifest['source_split_seeds'][split]+j
            rng = np.random.default_rng(seed+80000000)
            sim.peak_seconds = float(rng.uniform(1024., 3072.))
            signal, links, metadata = sim.signal(draw_parameters(seed))
            detector_peak = metadata['detector_energy_peak_index']/sim.fs
            duration = float(rng.uniform(33., 53.))
            # Half targeted near the detector peak; half uniform over the record.
            start = detector_peak-rng.uniform(0., duration) if j%2 == 0 else rng.uniform(0., sim.n/sim.fs-duration-150.)
            start = float(np.clip(start, 0, sim.n/sim.fs-duration-150.))
            link = 'all' if j%4 == 0 else int(rng.choice(sim.links))
            events = [dict(link=link, start_s=start, duration_s=duration)]
            if j%4 == 3:
                events.append(dict(link=int(rng.choice(sim.links)), start_s=start+200., duration_s=1.))
            _, _, valid = sim.masks(events)
            noise = sim.noise(seed+100000000)
            observed = np.where(valid, signal+noise, 0)
            features = make_features(observed, valid, kernel, scale)
            metadata.update(source_seed=seed, events=events, split=split,
                            missing_seconds_by_channel=((~valid).sum(axis=-1)/sim.fs).tolist())
            target = (signal/scale[:, None]).astype(np.float32)
            path = out/split/f'{j:05d}.npz'
            np.savez_compressed(path, features=features, target=target, valid=valid,
                                metadata=json.dumps(metadata))
            entries.append(dict(file=path.name, source_seed=seed,
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            print(f'{split} {j+1}/{count}: M={metadata["mass_source"]:.3g}, gap={duration:.1f}s', flush=True)
        manifest['splits'][split] = entries
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    manifest['complete'] = True
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--train', type=int, default=32)
    p.add_argument('--val', type=int, default=8)
    p.add_argument('--test', type=int, default=0)
    p.add_argument('--calibration', type=int, default=128)
    a = p.parse_args()
    generate(a.output, a.train, a.val, a.test, a.calibration)
