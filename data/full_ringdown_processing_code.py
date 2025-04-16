
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter

# Load data
f = np.loadtxt("f.txt")
h = np.loadtxt("h.txt", dtype=complex)

# Construct full spectrum (Hermitian symmetry)
h_neg = np.conj(h[1:-1][::-1])
h_full = np.concatenate([h, h_neg])

# Sampling parameters
N = len(h_full)
df = f[1] - f[0]
fs = df * N
dt = 1 / fs
time = np.arange(N) / fs

# Center the signal in time using frequency-domain phase shift
t0_shift = N / (2 * fs)
delay_phase = np.exp(-2j * np.pi * f * t0_shift)
h_shifted = h * delay_phase
h_neg_shifted = np.conj(h_shifted[1:-1][::-1])
h_full_shifted = np.concatenate([h_shifted, h_neg_shifted])
h_time_centered = np.fft.ifft(h_full_shifted).real

# Extract t > 0 part only to remove symmetry and transient
half_index = N // 2
h_post = h_time_centered[half_index:]
time_post = time[half_index:]

# High-pass FIR filter to smooth
cutoff = 1e-4  # Hz
numtaps = 51
fir_coeff = firwin(numtaps, cutoff, pass_zero=False, fs=fs)
h_post_smooth = lfilter(fir_coeff, [1.0], h_post)

# Save to file
final_data = np.column_stack((time_post, h_post_smooth))
np.savetxt("cleaned_ringdown_signal.txt", final_data, header="Time(s) Amplitude", fmt="%.10e")

# Plot result
plt.figure(figsize=(10, 4))
plt.plot(time_post, h_post_smooth, label='Smoothed Ringdown (t > 0)', color='purple')
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("One-sided Ringdown Signal (Transient Fully Removed)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
