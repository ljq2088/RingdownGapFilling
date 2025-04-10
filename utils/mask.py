import torch
from config.config import Config
def generate_continuous_mask(batch_size, signal_length,gap_size):
    """
    为每个批次中的信号生成一个在随机位置的连续布尔类型掩码。

    参数:
    - batch_size: 批次大小
    - signal_length: 每个信号的长度
    - mask_length: 掩码的连续长度

    返回:
    - masks: 一个布尔型掩码张量，形状为 [batch_size, signal_length]
    """
    masks = torch.ones((batch_size, signal_length), dtype=torch.bool)  # 生成一个全是True的布尔掩码
    
    # 随机生成每个批次的掩码起始位置
    #print(gap_size,signal_length)
    #gap_start = torch.randint(0, signal_length - (Config.signal_to_gap_length_ratio-2)*gap_size + 1, (batch_size,))

    for i in range(batch_size):
        gap_start = torch.randint(0, signal_length - (Config.signal_to_gap_length_ratio-2)*gap_size + 1, (batch_size,))
        masks[i, gap_start[i]:gap_start[i] + gap_size] = False  # 将连续的部分设为False

    return masks#,gap_start


def generate_continuous_mask_with_noise(batch_size, total_length,signal_length,gap_size):
    """
    为每个批次中的信号生成一个在随机位置的连续布尔类型掩码。

    参数:
    - batch_size: 批次大小
    - signal_length: 每个信号的长度
    - mask_length: 掩码的连续长度

    返回:
    - masks: 一个布尔型掩码张量，形状为 [batch_size, signal_length]
    """
    masks = torch.ones((batch_size, signal_length), dtype=torch.bool)  # 生成一个全是True的布尔掩码
    
    # 随机生成每个批次的掩码起始位置
    #print(gap_size,signal_length)
    #gap_start = torch.randint(0, signal_length - (Config.signal_to_gap_length_ratio-2)*gap_size + 1, (batch_size,))
    start=int(1/2*(total_length))
    for i in range(batch_size):
        gap_start = torch.randint(start, start+signal_length - (Config.signal_to_gap_length_ratio-2)*gap_size + 1, (batch_size,))
        masks[i, gap_start[i]:gap_start[i] + gap_size] = False  # 将连续的部分设为False

    return masks#,gap_start




