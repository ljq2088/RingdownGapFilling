"""Physical coverage and padding checks at fixed intrinsic/source parameters."""
import argparse, json
from pathlib import Path
import numpy as np
from .physics import Simulator
from .metrics import mismatch

def run(output):
    base = dict(mass_source=8e5, q=.7, redshift=2., inclination=1., longitude=.8,
                latitude=.3, polarization=.5, phase=.4)
    reference = Simulator(n=8192, peak_seconds=1024., pad_seconds=2048.)
    extended = Simulator(n=8192, peak_seconds=1024., pad_seconds=3072.)
    h1, _, m1 = reference.signal(base); h2, _, m2 = extended.signal(base)
    psd = reference.calibrate_psd(64)
    padding_mismatch = float(mismatch(h2, h1, psd))
    padding_relative_l2 = float(np.linalg.norm(h2-h1)/np.linalg.norm(h1))
    report = dict(padding_seconds=[2048., 3072.], padding_direct_mismatch=padding_mismatch,
                  padding_relative_l2=padding_relative_l2, prior_corners=[])
    simulator = Simulator(peak_seconds=3072.)
    for mass, z in [(1e5, 1.), (5e6, 6.)]:
        for q in [.3, 1.]:
            pars = dict(base, mass_source=mass, redshift=z, q=q)
            signal, _, metadata = simulator.signal(pars)
            report['prior_corners'].append(dict(mass_source=mass, redshift=z, q=q,
                         tau220_seconds=metadata['tau220_estimate_seconds'],
                         source_prehistory_seconds=metadata['prehistory_seconds'],
                         terminal_relative_amplitude=metadata['terminal_relative_amplitude'],
                         max_channel_amplitude=float(abs(signal).max())))
    report['padding_gate_passed'] = padding_mismatch < 1e-4 and padding_relative_l2 < .01
    Path(output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if not report['padding_gate_passed']:
        raise RuntimeError('Padding convergence gate failed; examine IMR placement and f_lower')

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    run(p.parse_args().output)
