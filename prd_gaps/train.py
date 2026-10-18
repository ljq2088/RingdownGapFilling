"""Validation-driven pilot training with atomic resumable checkpoints.

No test directory is read. Selection uses direct A/E mismatch on the detector
peak-to-six-damping-times window, reported alongside complete-record mismatch.
The damping-time estimate only defines evaluation/training windows.
"""
import argparse, hashlib, json, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .model import GapReconstructor, supervised_loss
from .metrics import mismatch

class Records(Dataset):
    def __init__(self, root, split):
        if split not in ('train', 'val'): raise ValueError('Training may read train/val only')
        self.records = []
        manifest = json.loads((Path(root)/'manifest.json').read_text())
        for entry in manifest['splits'][split]:
            path = Path(root)/split/entry['file']
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError(f'Dataset hash mismatch: {path.name}')
            with np.load(path) as a:
                self.records.append((a['features'], a['target'], a['valid'].astype(np.float32),
                                     json.loads(str(a['metadata']))))
        if not self.records: raise ValueError(f'Empty {split} set')
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        x, y, v, meta = self.records[idx]
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(v), idx

def window_mismatch(pred, target, psd, meta):
    n = target.shape[-1]; fs = meta['fs_hz']
    start = int(meta['detector_energy_peak_index'])
    stop = min(n, start+int(np.ceil(6*meta['tau220_estimate_seconds']*fs)))
    # An even interval preserves real-Nyquist quadrature conventions.
    stop -= (stop-start)%2
    if stop-start < 16: raise ValueError('Insufficient ringdown evaluation samples')
    freq_full = np.fft.rfftfreq(n, 1/fs)
    freq_crop = np.fft.rfftfreq(stop-start, 1/fs)
    local_psd = np.stack([np.interp(freq_crop, freq_full, channel) for channel in psd])
    return float(mismatch(pred[:, start:stop], target[:, start:stop], local_psd))

def save_atomic(state, path):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary); os.replace(temporary, path)

def run(args):
    torch.set_num_threads(4); torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    root = Path(args.data); out = Path(args.output)
    if out.exists() and not args.resume: raise FileExistsError('Use a fresh output directory or --resume')
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root/'manifest.json').read_text())
    if not manifest.get('complete'): raise ValueError('Dataset generation is incomplete')
    fingerprint = hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
    train = Records(root, 'train'); val = Records(root, 'val')
    with np.load(root/'calibration.npz') as calibration:
        psd = calibration['psd']/calibration['scale'][:, None]**2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GapReconstructor(layers=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=4)
    psd_tensor = torch.from_numpy(psd.astype(np.float32)).to(device)
    best, stale, first = float('inf'), 0, 0
    if args.resume:
        checkpoint = torch.load(out/'last.pt', map_location=device, weights_only=False)
        if checkpoint['manifest_sha256'] != fingerprint: raise ValueError('Dataset changed since checkpoint')
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best, stale, first = checkpoint['best'], checkpoint['stale'], checkpoint['epoch']+1
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        if device == 'cuda': torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint['cuda_rng']])
    trainloader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valloader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    config = dict(vars(args), manifest_sha256=fingerprint, parameters=sum(p.numel() for p in model.parameters()),
                  source_code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in Path(__file__).parent.glob('*.py')},
                  selection_metric='mean direct A/E mismatch: detector peak through six tau220',
                  data_role='pilot; no independent test evaluated', device=device)
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    if first >= args.epochs:
        raise ValueError('Resume epoch budget must exceed the completed epoch count')
    for epoch in range(first, args.epochs):
        tic = time.perf_counter(); model.train(); total = 0.
        for x, y, valid, indices in trainloader:
            x, y, valid = x.to(device), y.to(device), valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = supervised_loss(prediction, y, valid, psd_tensor)
            # Give the merger/ringdown window equal sample influence despite long IMR records.
            local = []
            for b, index in enumerate(indices.tolist()):
                meta = train.records[index][3]; start = meta['detector_energy_peak_index']
                stop = min(y.shape[-1], start+int(np.ceil(6*meta['tau220_estimate_seconds']*meta['fs_hz'])))
                yr = y[b, :, start:stop]; pr = prediction[b, :, start:stop]
                local.append(torch.log1p((pr-yr).square().mean()/yr.square().mean().clamp_min(1e-8)))
            loss = loss+torch.stack(local).mean()
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss; last checkpoint preserved')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); total += float(loss.detach())*len(indices)
        model.eval(); full, ringdown = [], []
        with torch.no_grad():
            for x, y, valid, indices in valloader:
                prediction = model(x.to(device)).cpu().numpy(); targets = y.numpy()
                full.extend(np.atleast_1d(mismatch(prediction, targets, psd)).tolist())
                for b, index in enumerate(indices.tolist()):
                    ringdown.append(window_mismatch(prediction[b], targets[b], psd, val.records[index][3]))
        score = float(np.mean(ringdown)); scheduler.step(score)
        improved = score < best-args.min_delta
        if improved: best, stale = score, 0
        else: stale += 1
        record = dict(epoch=epoch+1, train_loss=total/len(train), val_full_mean=float(np.mean(full)),
                      val_ringdown_mean=score, val_ringdown_median=float(np.median(ringdown)),
                      val_ringdown_p95=float(np.quantile(ringdown, .95)), seconds=time.perf_counter()-tic,
                      learning_rate=optimizer.param_groups[0]['lr'], best=best, stale=stale)
        with (out/'history.jsonl').open('a') as handle: handle.write(json.dumps(record)+'\n')
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     epoch=epoch, best=best, stale=stale, manifest_sha256=fingerprint,
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else [])
        save_atomic(state, out/'last.pt')
        if improved: save_atomic(state, out/'best.pt')
        print(json.dumps(record), flush=True)
        if stale >= args.patience: break
    (out/'status.json').write_text(json.dumps(dict(status='completed', epochs=epoch+1, best_validation=best,
                                    stop_reason='early stopping' if stale >= args.patience else 'epoch budget'))+'\n')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True); p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=100); p.add_argument('--patience', type=int, default=12)
    p.add_argument('--min-delta', type=float, default=1e-4); p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--batch-size', type=int, default=2); p.add_argument('--seed', type=int, default=126)
    p.add_argument('--resume', action='store_true')
    run(p.parse_args())
