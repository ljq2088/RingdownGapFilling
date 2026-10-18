"""Direct A/E waveform similarity, with fixed PSD and no alignment search."""
import numpy as np

def mismatch(prediction, target, psd, floor_fraction=1e-4):
    p, t = np.fft.rfft(prediction, axis=-1), np.fft.rfft(target, axis=-1)
    psd = np.asarray(psd)
    floor = np.median(psd[:, 1:], axis=-1)[:, None]*floor_fraction
    weight = 1/np.maximum(psd, floor)
    weight[:, 0] = 0; weight[:, -1] *= .5
    num = np.real(p*t.conj()*weight).sum(axis=(-2, -1))
    norm = np.sqrt((abs(p)**2*weight).sum(axis=(-2, -1))*(abs(t)**2*weight).sum(axis=(-2, -1)))
    return 1-num/np.maximum(norm, np.finfo(float).tiny)
