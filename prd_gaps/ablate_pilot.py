"""Paired validation stress tests for masks and delayed neighboring information."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from .train import Records, window_mismatch
from .model import GapReconstructor
from .physics import Simulator, ARM_SECONDS
from .masks import erode_valid
from .preprocess import make_features

def run(data, checkpoint, output):
    torch.set_num_threads(4)
    records = Records(data, 'val')
    with np.load(Path(data)/'calibration.npz') as calibration:
        scale, kernel = calibration['scale'], calibration['kernel']
        psd = calibration['psd']/scale[:, None]**2
    sim = Simulator()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GapReconstructor(layers=4).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state['model']); model.eval()
    guard = int(np.ceil(7*ARM_SECONDS*sim.fs))+sim.order+2
    results = []
    for stored_features, target, valid_float, meta in records.records:
        valid = valid_float.astype(bool)
        signal = target*scale[:, None]
        complete = signal+sim.noise(meta['source_seed']+100000000)
        guarded = erode_valid(np.pad(valid, ((0, 0), (guard, guard)), constant_values=True), guard)[:, guard:-guard]
        simple = np.ones(valid.shape, bool)
        for event in meta['events']:
            a = int(np.floor(event['start_s']*sim.fs))
            b = int(np.ceil((event['start_s']+event['duration_s'])*sim.fs))
            simple[:, a:b] = False
        start = meta['detector_energy_peak_index']
        stop = min(sim.n, start+int(6*meta['tau220_estimate_seconds']*sim.fs))
        row = dict(source_seed=meta['source_seed'], ringdown_has_gap=bool((~valid[:, start:stop]).any()))
        for name, mask in [('post_tdi_interval', simple), ('propagated_link', valid), ('extended_guard', guarded)]:
            # Use the exact stored input for the primary case; regeneration only
            # supplies observed noise for counterfactual availability conditions.
            features = stored_features if name == 'propagated_link' else make_features(complete, mask, kernel, scale)
            with torch.no_grad():
                prediction = model(torch.from_numpy(features)[None].to(device))[0].cpu().numpy()
            row[name] = window_mismatch(prediction, target, psd, meta)
        results.append(row)
    summary = {}
    for subset, rows in [('all', results), ('ringdown_has_gap', [r for r in results if r['ringdown_has_gap']])]:
        summary[subset] = dict(n=len(rows), **{name: float(np.mean([r[name] for r in rows]))
                             for name in ['post_tdi_interval', 'propagated_link', 'extended_guard']})
    report = dict(role='Paired validation stress test; model trained on propagated-link masks',
                  guard_seconds=guard/sim.fs, best_epoch=state['epoch']+1,
                  summary=summary, samples=results)
    Path(output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--data', required=True)
    p.add_argument('--checkpoint', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); run(a.data, a.checkpoint, a.output)
