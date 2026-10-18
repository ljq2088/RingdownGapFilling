"""Paired missing-data and covariance diagnostics, without model selection."""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.linalg import toeplitz
from .physics import Simulator, to_ae, tdi_from_links
from .masks import edge_taper, erode_valid
from .covariance import covariance_lags

def covariance_offdiagonal_fraction(covariance, weights):
    weighted = covariance*weights[:, None]*weights[None, :]
    # F C F^H, with NumPy's unnormalized forward Fourier matrix.
    spectral = np.fft.ifft(np.fft.fft(weighted, axis=0), axis=1)*len(weights)
    spectral = spectral[1:len(weights)//2, 1:len(weights)//2]
    all_power = np.sum(abs(spectral)**2)
    return float((all_power-np.sum(abs(np.diag(spectral))**2))/all_power)

def run(output):
    rng = np.random.default_rng(531)
    sim = Simulator(n=8192, peak_seconds=1024.)
    sim.response.get_projections(np.zeros(sim.parent_n, complex), .8, .3, t0=sim.pad/sim.fs)
    # Arbitrary link streams exercise the operator independently of any waveform.
    random_links = rng.normal(size=(6, sim.parent_n))
    complete = to_ae(tdi_from_links(sim.response, random_links))[:, sim.crop]
    rows = []
    for link in list(map(int, sim.links))+['all']:
        for duration in [.5, 33.]:
            events = [dict(link=link, start_s=1500., duration_s=duration)]
            lv, _, valid = sim.masks(events)
            replacement = random_links.copy(); replacement[~lv] = 100*rng.normal(size=(~lv).sum())
            modified = to_ae(tdi_from_links(sim.response, replacement))[:, sim.crop]
            delta = abs(modified-complete)
            error = float(delta[valid].max()/max(abs(complete).max(), 1e-30))
            assert error < 1e-10
            rows.append(dict(link=link, input_gap_seconds=duration,
                             missing_AE_seconds=((~valid).sum(axis=-1)/sim.fs).tolist(),
                             retained_relative_error=error))
    # Exact finite covariance, using the analytic stationary secondary-noise PSD.
    length = 512
    covariance = toeplitz(covariance_lags(sim.psd[0], sim.fs, length))
    covariance /= covariance[0, 0]
    valid = np.ones(length, bool); valid[220:286] = False
    tapered = edge_taper(valid[None, :], 32)[0]
    fractions = {name: covariance_offdiagonal_fraction(covariance, weights)
                 for name, weights in [('complete', np.ones(length)), ('hard_gap', valid),
                                       ('tapered_gap', tapered)]}
    # Mask-only echo stress control expands both channel masks by seven arm
    # delays plus the interpolation guard. Its effect requires paired evaluation.
    from .physics import ARM_SECONDS
    guard = int(np.ceil(7*ARM_SECONDS*sim.fs))+sim.order+2
    _, _, primary = sim.masks([dict(link=12, start_s=1500., duration_s=33.)])
    guarded = erode_valid(np.pad(primary, ((0, 0), (guard, guard)), constant_values=True), guard)[:, guard:-guard]
    assert np.all(guarded <= primary)
    report = dict(link_replacement_cases=rows, exact_spectral_covariance_offdiagonal_fraction=fractions,
                  echo_guard_samples=guard,
                  primary_missing_AE_seconds=((~primary).sum(axis=-1)/sim.fs).tolist(),
                  guarded_missing_AE_seconds=((~guarded).sum(axis=-1)/sim.fs).tolist(),
                  covariance_note='Finite colored-noise windows already couple bins; tapering changes covariance and effective exposure.')
    Path(output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    run(p.parse_args().output)
