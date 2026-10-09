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


from pytorch_wavelets import DWT1D, IDWT1D
import torch

def wavelet_bandpass_gpu(data_tensor, nFreq=8, wave='db4', device=None):
    """
    批量化的带通分解，输入输出形状均带有 batch_size：
      - 输入：data_tensor，shape=(batch_size, L)
      - 输出：bands_tensor，shape=(batch_size, nFreq, L)

    内部逻辑：
      1. 把 (batch_size, L) → (batch_size, 1, L)，送入 DWT1D 做多层分解（J=nFreq）。
      2. DWT 得到 cA_J 和 cD_list（长度为 J 的 tuple），每个都带有 batch 维。
      3. 对于第 i 个频带：让“只保留这一层的系数”，其他层置零，然后用 IDWT1D 重构得到 (batch_size, 1, L_recon)。
      4. 把通道维 squeeze 掉，裁剪/填充到原始长度 L，得到 (batch_size, L)。
      5. 最后把这 nFreq 条时域子带信号在通道维上 stack，输出 (batch_size, nFreq, L)。
    """
    # data_tensor: (batch_size, L)
    batch_size, L = data_tensor.shape

    # 先搬到 device 上，并加 channel 维：(batch_size, 1, L)
    device = data_tensor.device if device is None else torch.device(device)
    x = data_tensor.to(device).unsqueeze(1)

    # 分解层数 J = nFreq
    J = nFreq
    dwt = DWT1D(wave=wave, J=J).to(device)
    idwt = IDWT1D(wave=wave).to(device)

    # 多层 DWT
    coeffs = dwt(x)
    cA = coeffs[0]         # (batch_size, 1, L/(2**J))
    cD_list = coeffs[1]    # tuple：长度 J，每个元素 shape=(batch_size, 1, L/(2**j))

    bands = []
    for band_idx in range(nFreq):
        if band_idx == 0:
            # 只保留第 J 层的低频近似 cA_J，其他细节全置零
            keep_cA = cA.clone()
            zero_cD = tuple(torch.zeros_like(cd) for cd in cD_list)
            this_coeffs = (keep_cA, zero_cD)
        else:
            # 只保留对应的第 i 个高频细节层 cD_{J-i+1}，其他全部置零
            zero_cA = torch.zeros_like(cA)
            keep_cD = []
            for idx, cd in enumerate(cD_list):
                # cD_list 顺序是 (cD_J, cD_{J-1}, ..., cD_1)
                # 实际层号 j = J - idx
                if (J - idx) == band_idx:
                    keep_cD.append(cd.clone())
                else:
                    keep_cD.append(torch.zeros_like(cd))
            this_coeffs = (zero_cA, tuple(keep_cD))

        # 用 IDWT 重构 (batch_size, 1, L_recon)
        recon = idwt(this_coeffs)
        # (batch_size, 1, L_recon) → (batch_size, L_recon)
        recon = recon.squeeze(1)

        # 对齐回原来的长度 L：裁剪或 pad
        Lr = recon.size(-1)
        if Lr > L:
            recon = recon[..., :L]
        elif Lr < L:
            pad = L - Lr
            recon = torch.nn.functional.pad(recon, (0, pad))

        # recon 现在是 (batch_size, L)
        bands.append(recon)

    # 把所有子带在“通道”维度上叠起来：list length=nFreq，每个 (batch_size, L)
    # → tensor shape=(batch_size, nFreq, L)
    bands_tensor = torch.stack(bands, dim=1)
    return bands_tensor


def wavelet_reconstruct_gpu(bands_tensor):
    """
    批量化的多频道还原，输入输出形状均带有 batch_size：
      - 输入：bands_tensor，shape=(batch_size, nFreq, L)
      - 输出：recon，shape=(batch_size, 1, L)

    因为 bands_tensor 每条通道已经是“时域子带波形”，所以直接在 nFreq 通道上求和即可重构。
    """
    # bands_tensor: (batch_size, nFreq, L)
    recon = bands_tensor.sum(dim=1, keepdim=True)  # (batch_size, 1, L)
    return recon
