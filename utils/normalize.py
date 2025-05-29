import numpy as np
import torch

def normalize(signals: torch.Tensor, eps=1e-8):
    """
    对输入信号按样本维度做 normalize。
    输入：
        signals: Tensor, shape = [batch_size, seq_len]
    输出：
        normalized_signals: shape = [batch_size, seq_len]
        means: shape = [batch_size,]
        stds: shape = [batch_size,]
    """
    means = signals.mean(dim=1, keepdim=True)  # [B, 1]
    stds = signals.std(dim=1, keepdim=True)    # [B, 1]
    
    stds = stds.clamp(min=eps)  # 避免除以 0

    normalized = (signals - means) / stds

    return normalized, means.squeeze(1), stds.squeeze(1)
