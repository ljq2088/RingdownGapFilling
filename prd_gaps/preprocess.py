"""Noise-calibrated finite whitening with explicit filter-support validity."""
import numpy as np
from scipy.signal import fftconvolve
from .masks import erode_valid

def design_fir(psd, fs, radius=32, floor_fraction=1e-4):
    """Regularized, truncated zero-phase FIR; an approximate whitening feature.

    The PSD must come from independent complete noise records. Flooring TDI
    nulls avoids unbounded inverse gain; truncation error is measured separately.
    """
    psd = np.asarray(psd, float)
    if psd.ndim != 2 or psd.shape[0] != 2 or not np.isfinite(psd).all() or (psd < 0).any():
        raise ValueError('Expected finite nonnegative A/E PSD')
    n = 2*(psd.shape[-1]-1)
    if n <= 2*radius+1:
        raise ValueError('PSD grid too short for filter')
    # Input is the one-sided periodogram expectation: DC/Nyquist bins have
    # already retained their single-quadrature power during calibration.
    variance = fs/n*psd.sum(axis=-1)
    scale = np.sqrt(variance)
    if (scale <= 0).any(): raise ValueError('Noise variance must be positive')
    normalized = psd/scale[:, None]**2
    floor = np.median(normalized[:, 1:], axis=-1)*floor_fraction
    response = 1/np.sqrt(np.maximum(normalized, floor[:, None]))
    response[:, 0] = 0
    kernel = np.fft.fftshift(np.fft.irfft(response, n=n, axis=-1), axes=-1)
    kernel = kernel[:, n//2-radius:n//2+radius+1]*np.kaiser(2*radius+1, 8)[None, :]
    transfer = np.fft.rfft(kernel, n=n, axis=-1)
    outvar = fs/n*(normalized*abs(transfer)**2).sum(axis=-1)
    kernel /= np.sqrt(outvar)[:, None]
    return kernel, scale

def make_features(data, valid, kernel, scale):
    data, valid = np.asarray(data), np.asarray(valid, bool)
    if data.shape != valid.shape or data.shape[0] != 2:
        raise ValueError('A/E data and individual masks must have equal shapes')
    # np.where removes arbitrary hidden values (including NaNs) before filtering.
    raw = np.where(valid, data, 0)/np.asarray(scale)[:, None]
    support = erode_valid(valid, kernel.shape[-1]//2)
    white = np.stack([fftconvolve(x, k, mode='same') for x, k in zip(raw, kernel)])
    white = np.where(support, white, 0)
    from scipy.ndimage import distance_transform_edt
    distances = []
    for row in valid:
        inside = distance_transform_edt(np.pad(row, 1))[1:-1]
        missing = distance_transform_edt(np.pad(~row, 1))[1:-1]
        distances.append(np.tanh((inside-missing)/64))
    features = np.concatenate((raw, white, valid.astype(float), support.astype(float), distances))
    return features.astype(np.float32)
