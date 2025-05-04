import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config.config import Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def combine_segments(segments, segment_length=Config.segment_length, signal_length=Config.signal_length, overlap=Config.overlap):
    """
    将分段的信号重新组合为原始信号，并处理重叠部分。
    
    参数:
    - segments: 输入形状为 (batch_size, channels, num_segments, segment_length) 的张量
    - segment_length: 每个分段的长度，默认为 64
    - signal_length: 重组后的信号总长度，默认为 1056
    - overlap: 重叠率，默认为 50%

    返回:
    - 重组后的信号，形状为 (batch_size, channels, signal_length)
    """
    batch_size, channels, num_segments, _ = segments.shape
    step_size = int(segment_length * (1 - overlap))  # 步长，重叠 50% 的话，步长是 segment_length 的一半

    # 初始化输出张量，用于存储重新拼接后的信号
    output = torch.zeros((batch_size, channels, signal_length), dtype=segments.dtype)

    # 初始化一个计数器张量，用于记录每个位置被覆盖的次数
    counter = torch.zeros((batch_size, channels, signal_length), dtype=segments.dtype)

    # 将每个 segment 拼接到输出张量中
    for i in range(num_segments):
        start = i * step_size  # 计算每个分段的起始位置
        end = start + segment_length
        output=output.to(device)
        segments=segments.to(device)
        counter=counter.to(device)
        # 将当前分段添加到输出张量
        output[:, :, start:end] += segments[:, :, i, :]
        
        # 计数器记录每个位置被覆盖的次数
        counter[:, :, start:end] += 1

    # 处理重叠区域：将那些被多次覆盖的部分除以覆盖次数（即重叠部分除以 2）
    output = output / counter

    return output