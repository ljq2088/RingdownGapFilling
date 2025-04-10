import numpy as np
from scipy.fft import fft

def calculate_mismatch(signal1, signal2, psd=None, delta_t=1.0):
    """
    计算两个归一化信号的 mismatch.
    
    参数:
        signal1 (array): 第一个信号序列
        signal2 (array): 第二个信号序列
        psd (array): 噪声功率谱密度, 如果没有提供, 默认为1
        delta_t (float): 采样间隔, 默认值为1.0
    
    返回:
        float: mismatch 值
    """
    # 计算傅里叶变换
    fft1 = fft(signal1)
    fft2 = fft(signal2)
    
    # 频域积分区间（Nyquist频率）
    N = len(signal1)
    freqs = np.fft.fftfreq(N, delta_t)
    df = freqs[1] - freqs[0]  # 频率间隔
    
    # 如果没有提供 PSD，假设为1
    if psd is None:
        psd = np.ones_like(freqs)
    
    # 计算内积
    inner_product = np.sum((fft1 * fft2.conj() / psd).real) * df
    norm1 = np.sum((fft1 * fft1.conj() / psd).real) * df
    norm2 = np.sum((fft2 * fft2.conj() / psd).real) * df
    
    # 归一化后的 overlap
    overlap = inner_product / np.sqrt(norm1 * norm2)
    
    # 计算 mismatch
    mismatch = 1 - overlap
    return mismatch.real  # 确保返回实数部分

# # 测试信号
# signal1 = np.sin(2 * np.pi * np.linspace(0, 1, 1024))
# signal2 = np.sin(2 * np.pi * np.linspace(0, 1, 1024) + 0.1)  # 相位偏移

# # 计算 mismatch
# mismatch = calculate_mismatch(signal1, signal2)
# print(f"Mismatch: {mismatch:.6f}")
