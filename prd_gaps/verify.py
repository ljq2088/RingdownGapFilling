"""Run with python -m prd_gaps.verify --output /path/to/report.json."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from scipy.signal import fftconvolve
from .masks import edge_taper
from .preprocess import design_fir, make_features
from .covariance import covariance_lags, observed_quadratic
from .model import GapReconstructor, supervised_loss
from .metrics import mismatch

def verify(output, physics=True):
    started = time.perf_counter(); torch.set_num_threads(4); torch.manual_seed(126)
    rng = np.random.default_rng(126)
    n, fs = 2048, 2.
    f = np.fft.rfftfreq(n, 1/fs)
    psd = np.stack((1/(.02+f)**2, 2/(.02+f)**2))
    kernel, scale = design_fir(psd, fs, radius=24)
    valid = np.ones((2, n), bool); valid[0, 600:780] = False; valid[1, 650:840] = False
    data = rng.normal(size=(2, n))
    features = make_features(data, valid, kernel, scale)
    changed = data.copy(); changed[~valid] = np.nan
    assert np.array_equal(features, make_features(changed, valid, kernel, scale))
    white = np.stack([fftconvolve(x/s, k, mode='same') for x, s, k in zip(data, scale, kernel)])
    support = features[6:8].astype(bool)
    fir_error = float(np.max(abs(white[support]-features[2:4][support])))
    assert fir_error < 1e-6
    # Gap-edge leakage for a smooth low-frequency signal. Diagnostic, not a
    # statement that tapering recovers missing data or restores stationarity.
    smooth = np.stack([np.sin(2*np.pi*.03125*np.arange(n)/fs)]*2)
    rectangular = np.where(valid, smooth, 0)
    tapered = rectangular*edge_taper(valid, 48)
    high = f > .2
    high_rect = float((abs(np.fft.rfft(rectangular))[:, high]**2).sum())
    high_taper = float((abs(np.fft.rfft(tapered))[:, high]**2).sum())
    assert high_taper < high_rect
    # Noise covariance reference: dense exact marginal quadratic has mean dof.
    m, crop = 2048, 96
    noise_psd = 1/(.03+np.fft.rfftfreq(m, 1/fs))**2; noise_psd[0] = 0
    lags = covariance_lags(noise_psd, fs, crop)
    vm = np.ones(crop, bool); vm[30:50] = False
    values = []
    for _ in range(256):
        z = (rng.normal(size=m//2+1)+1j*rng.normal(size=m//2+1))*np.sqrt(m*fs*noise_psd/4)
        z[0] = 0; z[-1] = rng.normal()*np.sqrt(m*fs*noise_psd[-1]/2)
        noise = np.fft.irfft(z, n=m)[:crop]
        values.append(observed_quadratic(noise, vm, lags))
    mean_quad = float(np.mean(values)); dof = int(vm.sum())
    assert abs(mean_quad/dof-1) < .05
    # All standard attention projections must be used by an actual backward pass.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GapReconstructor(layers=4).to(device)
    x = torch.from_numpy(features)[None].to(device)
    target = torch.from_numpy((smooth/scale[:, None]).astype(np.float32))[None].to(device)
    v = torch.from_numpy(valid.astype(np.float32))[None].to(device)
    p = torch.from_numpy((psd/scale[:, None]**2).astype(np.float32)).to(device)
    prediction = model(x)
    loss = supervised_loss(prediction, target, v, p); loss.backward()
    grads = {name: float(param.grad.abs().sum()) if param.grad is not None else None
             for name, param in model.named_parameters() if 'self_attn' in name}
    assert all(value is not None and np.isfinite(value) and value > 0 for value in grads.values())
    qkv = [[float(block.abs().sum()) for block in layer.self_attn.in_proj_weight.grad.chunk(3)]
           for layer in model.attention.layers]
    assert all(value > 0 for layer in qkv for value in layer)
    assert prediction.shape == target.shape and torch.isfinite(prediction).all()
    assert float(mismatch(smooth, smooth, psd)) < 1e-12
    report = dict(fir_valid_support_max_error=fir_error, hidden_sample_invariance=True,
                  taper_high_frequency_power_ratio=high_taper/high_rect,
                  marginal_covariance_mean_quadratic=mean_quad, retained_dof=dof,
                  attention_gradients=grads, qkv_gradient_norms=qkv,
                  parameters=sum(p.numel() for p in model.parameters()),
                  initial_synthetic_loss=float(loss.detach()), device=device)
    if physics:
        from .physics import Simulator, to_ae, tdi_from_links
        simulator = Simulator(n=8192, peak_seconds=1024.)
        pars = dict(mass_source=8e5, q=.7, redshift=2., inclination=1., longitude=.8,
                    latitude=.3, polarization=.5, phase=.4)
        events = [dict(link=12, start_s=1030., duration_s=33.),
                  dict(link=23, start_s=1500., duration_s=1.)]
        example = simulator.example(126, events, pars)
        original = example['links']
        lv = example['link_valid']
        reference = to_ae(tdi_from_links(simulator.response, original))[:, simulator.crop]
        replacement = original.copy()
        replacement[~lv] = rng.normal(size=(~lv).sum())*np.max(abs(original))*10
        modified = to_ae(tdi_from_links(simulator.response, replacement))[:, simulator.crop]
        mask = example['valid']
        relative = float(np.max(abs(reference[mask]-modified[mask]))/np.max(abs(reference)))
        affected = float(np.max(abs(reference-modified))/np.max(abs(reference)))
        assert relative < 1e-10 and affected > 1e-3, (relative, affected)
        assert np.all(example['observation'][~mask] == 0)
        assert np.array_equal(example['observation'][mask], (example['signal']+example['noise'])[mask])
        calibration = simulator.calibrate_psd(count=64)
        k, s = design_fir(calibration, simulator.fs, radius=32)
        empirical_variance = np.mean([np.mean(simulator.noise(90000000+j)**2, axis=-1)
                                      for j in range(64)], axis=0)
        rms_relative_error = float(np.max(abs(s*s/empirical_variance-1)))
        assert rms_relative_error < 1e-10
        physical_features = make_features(example['observation'], mask, k, s)
        with torch.no_grad():
            actual_output = model(torch.from_numpy(physical_features)[None].to(device))
        assert torch.isfinite(actual_output).all()
        report['physics'] = dict(metadata=example['metadata'], links=simulator.links.tolist(),
                                 retained_replacement_relative_error=relative,
                                 unmasked_replacement_effect=affected,
                                 missing_seconds_by_channel=((~mask).sum(axis=-1)/simulator.fs).tolist(),
                                 calibration_noise_records=64,
                                 noise_variance_parseval_relative_error=rms_relative_error,
                                 fully_valid_fir_fraction=physical_features[6:8].mean(axis=-1).tolist())
        np.savez_compressed(Path(output).with_suffix('.npz'), **{key: example[key] for key in
                            ('signal', 'noise', 'observation', 'valid')}, psd=calibration,
                            features=physical_features, scale=s, kernel=k)
    report['seconds'] = time.perf_counter()-started
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--no-physics', action='store_true')
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    verify(args.output, physics=not args.no_physics)
