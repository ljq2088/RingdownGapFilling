"""Small-record exact reference for the marginal observed-data likelihood."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

def covariance_lags(psd, fs, n):
    """Parent stationary covariance, consistent with a zero-DC noise generator."""
    psd = np.asarray(psd, float).copy()
    m = 2*(len(psd)-1)
    if n > m//2: raise ValueError('Parent record must be at least twice the crop length')
    psd[0] = 0
    return (fs/2*np.fft.irfft(psd, n=m))[:n]

def observed_quadratic(residual, valid, lags):
    """r_O^T C_OO^{-1} r_O; no zero-filled Fourier-diagonal approximation.

    O(N^2) memory: reference use only, at most 2048 retained samples.
    """
    valid = np.asarray(valid, bool)
    idx = np.flatnonzero(valid)
    if len(idx) == 0 or len(idx) > 2048:
        raise ValueError('Reference requires 1..2048 retained samples')
    lags = np.asarray(lags)
    if len(lags) < len(valid): raise ValueError('Covariance lags do not cover the record')
    # Allocate only the observed submatrix, even if the parent grid is long.
    cov = lags[abs(idx[:, None]-idx[None, :])]
    norm = float(lags[0])
    factor = cho_factor(cov/norm, lower=True, check_finite=True)
    r = np.asarray(residual)[idx]/np.sqrt(norm)
    return float(r @ cho_solve(factor, r))
