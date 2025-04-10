from ssqueezepy import cwt, icwt, Wavelet
import numpy as np
import torch
def wavelet_bandpass(data, nFreq=8, wavelet_basis='morlet', numtaps = 6):
    # Determine the appropriate wavelet
    wavelet = Wavelet(wavelet_basis)
    
    # Compute CWT coefficients and scales
    scales = np.geomspace(6, 256, nFreq*4)
    # scales = np.linspace(4,256,nFreq*2)
    # scales = np.geomspace(7, 180, nFreq*4)
    

    # Number of scales
    num_scales = len(scales)

    # Divide scales into bands
    scale_indices = np.array_split(np.arange(num_scales), nFreq)

    # Reconstruct signals for each band
    reconstructed_signals = []
    transient_length = numtaps+1
    pad_length = transient_length  # 填充的长度等于 transient_length
    padded_signal = np.pad(data, pad_width=(pad_length, pad_length), mode='reflect')
    
    Wx, scales = cwt(padded_signal, wavelet=wavelet,scales = scales)
    #print(f"Wx shape: {Wx.shape}, scales shape: {scales.shape}")
    for indices in scale_indices:
        # Create a copy of Wx for band-specific modification
        Wx_band = np.zeros_like(Wx)
        # Set only the current band's indices to be non-zero
        Wx_band[indices, :] = Wx[indices, :]

        # Inverse CWT to reconstruct the band-specific signal
        reconstructed_signal = icwt(Wx_band, wavelet=wavelet,scales = scales)

        reconstructed_signal = reconstructed_signal[transient_length:-transient_length]
        if np.isnan(reconstructed_signal).any():
            print(reconstructed_signal[:10])
        # print the continous zero
        if np.isclose(reconstructed_signal,0).all():
            # print('close zero')
            pass
            # print(scale_indices[-5:])
        reconstructed_signals.append(reconstructed_signal)
    # print('done')
    return reconstructed_signals
def wavelet_reconstruct_from_channels(signals, wavelet_basis='morlet', scales=None):
    """
    将 8 个 channel 的小波变换结果合并为 1 个信号。
    
    参数:
    - signals: 输入形状为 (batch_size, channels, signal_length) 的信号张量
    - wavelet_basis: 小波基
    - scales: 小波的尺度，如果没有提供会自动生成

    返回:
    - 重构的信号，形状为 (batch_size, 1, signal_length)
    """
    batch_size, channels, signal_length = signals.shape

    # 确定小波基和尺度
    wavelet = Wavelet(wavelet_basis)
    if scales is None:
        scales = np.geomspace(6, 256, channels)  # 自动生成合适的尺度

    reconstructed_signals = []

    for i in range(batch_size):
        # 对每个信号的 channel 进行小波反变换
        signal_channels = signals[i].detach().cpu().numpy()  # 取出每个 batch 的信号并转成 numpy
        reconstructed_signal = icwt(signal_channels, wavelet=wavelet, scales=scales)
        reconstructed_signals.append(reconstructed_signal)

    # 转换为 (batch_size, 1, signal_length)
    reconstructed_signals = np.stack(reconstructed_signals, axis=0)
    return torch.tensor(reconstructed_signals, dtype=torch.float32).unsqueeze(1)