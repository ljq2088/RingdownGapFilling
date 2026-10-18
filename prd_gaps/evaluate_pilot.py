"""Validation-only pilot comparison. Never use this report as held-out evidence."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from .train import Records, window_mismatch
from .model import GapReconstructor
from .metrics import mismatch

def run(data, checkpoint, output):
    torch.set_num_threads(4)
    records = Records(data, 'val')
    with np.load(Path(data)/'calibration.npz') as calibration:
        psd = calibration['psd']/calibration['scale'][:, None]**2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GapReconstructor(layers=4).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state['model']); model.eval()
    rows = []
    for features, target, valid, metadata in records.records:
        with torch.no_grad():
            prediction = model(torch.from_numpy(features)[None].to(device))[0].cpu().numpy()
        baseline = features[:2]
        lowpass = sosfiltfilt(butter(6, .15, fs=metadata['fs_hz'], output='sos'), baseline, axis=-1)
        row = dict(source_seed=metadata['source_seed'], source_mass=metadata['mass_source'],
                   events=metadata['events'])
        for name, waveform in [('network', prediction), ('zero_fill', baseline), ('lowpass_0p15Hz', lowpass)]:
            row[name] = dict(full_mismatch=float(mismatch(waveform, target, psd)),
                             ringdown_mismatch=window_mismatch(waveform, target, psd, metadata))
        rows.append(row)
    report = dict(role='Validation pilot, reused for model selection; no independent test result',
                  best_epoch=state['epoch']+1, n=len(rows), samples=rows, summary={})
    for method in ['network', 'zero_fill', 'lowpass_0p15Hz']:
        report['summary'][method] = {}
        for metric in ['full_mismatch', 'ringdown_mismatch']:
            values = [r[method][metric] for r in rows]
            report['summary'][method][metric] = dict(mean=float(np.mean(values)),
                        median=float(np.median(values)), p95=float(np.quantile(values, .95)))
    Path(output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['summary'], indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--data', required=True)
    p.add_argument('--checkpoint', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); run(a.data, a.checkpoint, a.output)
