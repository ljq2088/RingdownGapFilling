import numpy as np

def compute_matched_filter_snr(s, h, PSD, samp_freq):
    N = len(s)
    df = samp_freq / N
    s_f = np.fft.fft(s)
    h_f = np.fft.fft(h)
    
    Sn = PSD[:N//2 + 1]
    numerator = s_f[:N//2 + 1] * np.conj(h_f[:N//2 + 1])
    matched_filter = 4 * numerator / Sn
    
    rho_t = np.fft.irfft(matched_filter, n=N)
    
    norm = np.sqrt(4 * np.sum(np.abs(h_f[:N//2 + 1])**2 / Sn) * df)
    rho_t /= norm
    
    return rho_t, np.max(np.abs(rho_t))
