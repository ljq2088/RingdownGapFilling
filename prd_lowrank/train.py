import os, time, json, copy, fcntl
from pathlib import Path
import numpy as np, torch
from prd_physical.common import Preprocess, write, sha
from prd_strategy.latent import CoefficientAdapter
from prd_strategy.loss import StrategyObjective
from .settings import *
from .common import status, check_sources, ram_guard, atomic_checkpoint
from .data import TrainingData, validation_data, reconstruct, validate, predict

def train_variant(kind, destination=OUT, count=TRAIN_N, max_epochs=MAX_EPOCHS, min_epochs=MIN_EPOCHS, patience=PATIENCE, stop_after=None):
    assert kind in VARIANTS
    destination = Path(destination)
    path = destination / kind
    path.mkdir(parents=True, exist_ok=True)
    with open(path / 'training.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (path / 'selected.json').exists():
            return json.loads((path / 'selected.json').read_text())
        torch.set_num_threads(2)
        torch.manual_seed(SEED)
        check_sources()
        ram_guard(path)
        basis = torch.tensor(np.load(PROBE / 'training_direction_basis.npz')['vectors'][:, :RANK], dtype=torch.float32)
        data = TrainingData(destination / 'cache', count, kind)
        val = validation_data(destination / 'cache')
        model = CoefficientAdapter(RANK)
        ema = copy.deepcopy(model)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=.5, patience=15, threshold=DELTA, threshold_mode='abs', min_lr=2e-6)
        cal = np.load(CAL)
        objective = StrategyObjective(Preprocess(cal['psd'], 'white'), cal, True)
        rng = torch.Generator().manual_seed(SEED+1)
        history, epoch, stale, best_epoch = [], 0, 0, 0
        best = validate(ema, val, basis)['score']
        best_state = copy.deepcopy(ema.state_dict())
        if (path / 'last.pth').exists():
            ck = torch.load(path / 'last.pth', map_location='cpu', weights_only=False)
            assert ck['count'] == count and ck['kind'] == kind
            model.load_state_dict(ck['raw']); ema.load_state_dict(ck['ema'])
            opt.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
            rng.set_state(ck['shuffle_rng']); torch.set_rng_state(ck['torch_rng'])
            history, epoch, stale = ck['history'], ck['epoch'], ck['stale']
            best, best_epoch, best_state = ck['best'], ck['best_epoch'], ck['best_state']
            status(path, 'resumed', epoch=epoch, best=best, best_epoch=best_epoch)
        else:
            atomic_checkpoint(path / 'best.pth', dict(state_dict=best_state, rank=RANK, kind=kind, epoch=0, score=best, basis=basis, backbone_sha=SNAPSHOT_SHA, count=count))
        for epoch in range(epoch+1, max_epochs+1):
            check_sources()
            ram_guard(path)
            t = time.monotonic()
            model.train(); total = 0.
            order = torch.randperm(count*VIEWS, generator=rng).numpy()
            for lo in range(0, len(order), BATCH):
                v = data.batch(order[lo:lo+BATCH])
                opt.zero_grad(set_to_none=True)
                y = reconstruct(model, v, basis)
                loss = objective(y, v['target'], v['mask'])
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite training loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise RuntimeError('nonfinite gradient')
                opt.step()
                with torch.no_grad():
                    for old, new in zip(ema.parameters(), model.parameters()):
                        old.lerp_(new, 1-EMA)
                total += float(loss.detach()) * len(v['target'])
            result = validate(ema, val, basis)
            if result['score'] < best-DELTA:
                best, best_epoch, stale = result['score'], epoch, 0
                best_state = copy.deepcopy(ema.state_dict())
                atomic_checkpoint(path / 'best.pth', dict(state_dict=best_state, rank=RANK, kind=kind, epoch=epoch, score=best, basis=basis, backbone_sha=SNAPSHOT_SHA, count=count))
            else:
                stale += 1
            scheduler.step(result['score'])
            row = dict(epoch=epoch, train_loss=total/len(order), validation=result, best=best, best_epoch=best_epoch, stale=stale, learning_rate=opt.param_groups[0]['lr'], epoch_seconds=time.monotonic()-t)
            history.append(row)
            atomic_checkpoint(path / 'last.pth', dict(raw=model.state_dict(), ema=ema.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(), shuffle_rng=rng.get_state(), torch_rng=torch.get_rng_state(), history=history, epoch=epoch, stale=stale, best=best, best_epoch=best_epoch, best_state=best_state, count=count, kind=kind))
            write(path / 'history.json', history)
            status(path, 'epoch_complete', **row)
            if stop_after is not None and epoch >= stop_after:
                return dict(interrupted_for_resume_test=True, epoch=epoch)
            if epoch >= min_epochs and stale >= patience:
                break
        ema.load_state_dict(best_state)
        result = validate(ema, val, basis)
        np.save(path / 'validation_prediction.npy', predict(ema, val, basis))
        selected = dict(passed=True, variant=kind, best_epoch=best_epoch, finished_epoch=epoch, validation=result, checkpoint_sha256=sha(path/'best.pth'), train_unique=count, training_observations_per_epoch=count*VIEWS, test_read=False, selected_weights='EMA .995', full_budget=(max_epochs==MAX_EPOCHS and count==TRAIN_N))
        write(path / 'selected.json', selected)
        status(path, 'completed', **selected)
        return selected

if __name__ == '__main__':
    import sys
    train_variant(sys.argv[1])
