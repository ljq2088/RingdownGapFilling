import numpy as np
import torch
def normalize(signal):
    signal=torch.tensor(signal)
    mean = signal.mean(dim=0)
    std = signal.std(dim=0)
    normalized_signal = (signal - mean) / std
    #normalized_masked_signal=(masked_signal-mean)/std
    return normalized_signal, mean,std