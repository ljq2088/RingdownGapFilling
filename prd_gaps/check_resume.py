"""Verify uninterrupted versus checkpoint-resumed optimization on pilot data."""
import argparse, json
from pathlib import Path
from types import SimpleNamespace
import torch
from .train import run

def check(data, output):
    root = Path(output)
    if root.exists(): raise FileExistsError(root)
    args = dict(data=data, epochs=2, patience=12, min_delta=1e-4, lr=2e-4,
                batch_size=4, seed=579, resume=False)
    run(SimpleNamespace(**args, output=str(root/'continuous')))
    args['epochs'] = 1
    run(SimpleNamespace(**args, output=str(root/'resumed')))
    args.update(epochs=2, resume=True)
    run(SimpleNamespace(**args, output=str(root/'resumed')))
    a = torch.load(root/'continuous/last.pt', map_location='cpu', weights_only=False)
    b = torch.load(root/'resumed/last.pt', map_location='cpu', weights_only=False)
    difference = max(float((a['model'][key]-b['model'][key]).abs().max()) for key in a['model'])
    # CUDA float32 reductions can differ at roundoff across fresh invocations.
    tolerance = 1e-5
    report = dict(max_parameter_difference=difference, tolerance=tolerance, passed=difference < tolerance,
                  completed_epochs=a['epoch']+1, same_best_score=abs(a['best']-b['best']) < tolerance)
    (root/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
    assert report['passed'] and report['same_best_score']

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--data', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); check(a.data, a.output)
