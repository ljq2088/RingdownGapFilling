import sys
import os
import math
from math import pi
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
from scipy.fftpack import fft, ifft
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import firwin2,welch
import numpy as np

def fftfilt(b, x):
    """Use FFT to apply FIR filter defined by b to signal x."""
    # 获取输入信号和滤波器的长度
    N_x = len(x)
    N_b = len(b)
    
    # 计算 FFT 的大小
    N = N_x + N_b - 1
    
    # 计算滤波器和输入信号的 FFT
    #X = np.fft.rfft(WGN) / np.sqrt(N)

    X = (np.random.randn(N//2 + 1) + 1j * np.random.randn(N//2 + 1)) / np.sqrt(2)

    B = np.fft.fft(b, N)
    
    # 乘以滤波器的频率响应
    Y = X * B
    
    # 计算逆 FFT 以获得滤波后的信号
    y = np.fft.ifft(Y)
    
    # 取出中间的部分，与输入信号长度相同
    start = (N_b - 1) // 2
    y = y[start:start + N_x]
    
    # 只取实部（虚部应该非常接近零）
    return np.real(y)


def stat_gauss_noise(n_samp,freq,PSD,flt_ord,samp_freq):
    sqrt_PSD=np.sqrt(PSD)
    sqrt_PSD[-1]=0
    #b=firwin2(flt_ord,freq/(samp_freq/2),sqrt_PSD)
    b=firwin2(flt_ord,freq/(samp_freq/2),sqrt_PSD)
    in_noise=np.random.randn(1,n_samp)
    # print(in_noise.shape,b.shape)
    # print(fftfilt(b,in_noise[0]).shape)
    return np.sqrt(samp_freq)*fftfilt(b,in_noise[0])
import numpy as np 
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries

def generate_noise_from_psd(N, freqVec, psd, sample_rate, num_noises=1,freq_range=[0,1], duration = None):
    #freqVec = np.linspace(0, sample_rate/2, len(psd))
    noises = np.zeros((num_noises, N))
    interpolated_asds = []
    for i in range(num_noises):
        WGN = np.random.randn(N)
        X = np.fft.rfft(WGN) / np.sqrt(N)
        asd = np.sqrt(psd)
        uneven = N % 2
        # Simulate the white noise of rFFT
        # X = (np.random.randn(N // 2 + 1 + uneven) + 1j * np.random.randn(N // 2 + 1 + uneven))
        
        selected_indices = np.where((freqVec >= freq_range[0]) & (freqVec <= freq_range[1]))[0]
        
        # Interpolate selected ASD values to match the length of X
        interp_asd = interp1d(freqVec[selected_indices],asd[selected_indices], kind='linear', bounds_error=False, fill_value="extrapolate")
        newFreqVec = np.fft.rfftfreq(N+uneven, d=1.0/sample_rate)
        interpolated_asd = interp_asd(newFreqVec)
        nonSelected_indices = np.where(~ ((newFreqVec> freq_range[0]) & (newFreqVec < freq_range[1])))[0]
        #interpolated_asd[nonSelected_indices] = 1e-30
        # interpolated_asd[interpolated_asd<1e-30] = 1e-30

        # Apply the random ASD to create colored noise
        # In order to keep the nSample equal to before
        Y_colored = X * interpolated_asd
        #y_colored = np.fft.irfft(Y_colored).real * np.sqrt(N*sample_rate)
        y_colored = np.fft.irfft(X * interpolated_asd, n=N) * np.sqrt(sample_rate)

        if uneven:
            y_colored = y_colored[:-1]
        
        noises[i, :] = y_colored 
        interpolated_asds.append(interpolated_asd)
        
    return noises, interpolated_asds
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import h5py

# Load LISA noise data
LISA_noise_data = h5py.File("/home/ljq/code/Ringdown_gap_filling/Proj/data/LISA_noise.mat", 'r')

# Extract noise data
nX = np.array(LISA_noise_data['nXVec'][:]).flatten()
nY = np.array(LISA_noise_data['nYVec'][:]).flatten()
nZ = np.array(LISA_noise_data['nZVec'][:]).flatten()

# Convert XYZ to A channel (simplified XYZ2AET function)
noise_A = (nZ - nX)/np.sqrt(2)

# Get sampling frequency
dt = 15  # assuming 15 seconds based on standard LISA data
fs = 1/dt

# Calculate PSD using Welch method
f_noise, psd_noise_A = signal.welch(noise_A, fs, nperseg=2**10)
# #print(psd_noise_A)
# # Plot the noise PSD
# plt.figure(figsize=(8, 5))
# plt.loglog(f_noise, np.sqrt(psd_noise_A), label='LISA Instrumental Noise (A channel)')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('ASD (1/√Hz)')
# plt.title('LISA Instrumental Noise PSD')
# plt.grid(True, which='both', linestyle='--', alpha=0.5)
# plt.legend()
# plt.show()
from scipy.interpolate import interp1d

# 自定义频率数组
freq = np.arange(0.001, 0.5, 0.001)

# 构建 PSD 插值函数（外推时自动填充极小值）
psd_interp_func = interp1d(f_noise, psd_noise_A,
                           kind='linear',
                           bounds_error=False,
                           fill_value=1e-40)