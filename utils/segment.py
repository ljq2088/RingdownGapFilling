import torch
import numpy as np
from ssqueezepy import cwt, Wavelet
from utils.wavelet import wavelet_bandpass
from config.config import Config


def segment_signal(transformed_signals, segment_length=Config.segment_length, overlap=Config.overlap):

    segmented_signals = []
    step_size = int(segment_length * (1 - overlap))

    for signal in transformed_signals:
        segments = []
        for start in range(0, len(signal) - segment_length + 1, step_size):
            segment = signal[start:start + segment_length]
            segments.append(segment)
        segmented_signals.append(np.array(segments))

    return segmented_signals
def segment_signal_gpu(transformed_signals, segment_length=Config.segment_length, overlap=Config.overlap):
    """
    对输入张量的最后一个维度进行分片操作。
    输入形状: (batch_size, channels, signal_length)
    输出形状: (batch_size, channels, num_segments, segment_length)

    :param transformed_signals: 输入信号，形状为 (batch_size, channels, signal_length)
    :param segment_length: 每个分片的长度
    :param overlap: 分片之间的重叠比例
    :return: 分片后的信号，形状为 (batch_size, channels, num_segments, segment_length)
    """
    batch_size, channels, signal_length = transformed_signals.shape
    step_size = int(segment_length * (1 - overlap))  # 计算步长
    num_segments = (signal_length - segment_length) // step_size + 1  # 计算分片数量

    # 初始化结果张量
    segmented_signals = torch.zeros((batch_size, channels, num_segments, segment_length), dtype=transformed_signals.dtype, device=transformed_signals.device)

    # 对最后一个维度进行分片
    for i in range(num_segments):
        start = i * step_size
        end = start + segment_length
        segmented_signals[:, :, i, :] = transformed_signals[:, :, start:end]

    return segmented_signals
def preprocess_signals(signals, segment_length=Config.segment_length, overlap=Config.overlap):
    processed_signals = []
    for signal in signals:
        transformed_signals = wavelet_bandpass(signal)
        segmented_signals = segment_signal(transformed_signals, segment_length=segment_length, overlap=overlap)
        processed_signals.append(segmented_signals)
    return processed_signals
