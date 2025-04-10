import torch
from torch.utils.data import Dataset
from config.config import Config
from utils.wavelet import wavelet_bandpass

# class GWSignalDataset(Dataset):
#     def __init__(self, signals,  conditions):
#         self.signals = signals 
#         #self.masked_signals = masked_signals
#         self.conditions = conditions

#     def __len__(self):
#         return len(self.signals)

#     def __getitem__(self, idx):
#         signal = self.signals[idx]
#         #masked_signal = self.masked_signals[idx]
#         condition = self.conditions[idx]
#         return torch.tensor(signal, dtype=torch.float32), torch.tensor(condition, dtype=torch.float32)
# class GWSignalDataset(Dataset):
#     def __init__(self, signals, conditions, nFreq=8, wavelet_basis='morlet', segment_length=Config.segment_length, overlap=Config.overlap):
#         self.signals = signals
#         self.conditions = conditions
#         self.nFreq = nFreq
#         self.wavelet_basis = wavelet_basis
#         self.segment_length = segment_length
#         self.stride = int(segment_length * (1 - overlap))  # 50% 重叠的步长

#     def __len__(self):
#         return len(self.signals)

#     def __getitem__(self, idx):
#         signal = self.signals[idx]  # 获取当前信号
#         condition = self.conditions[idx]
        

#         # 1. 使用 wavelet_bandpass 函数对信号进行小波变换，得到 8 个子频带信号
#         band_signals = wavelet_bandpass(signal, self.nFreq, wavelet_basis=self.wavelet_basis)

#         # 2. 对每个子频带信号进行分段
#         segmented_signals = []
#         for band_signal in band_signals:  # 对每个频带信号进行分段
#             segments = []
#             for i in range(0, len(band_signal) - self.segment_length + 1, self.stride):
#                 segment = band_signal[i:i + self.segment_length]
#                 segments.append(segment)
#             segments = torch.tensor(segments, dtype=torch.float32)  
#             segmented_signals.append(segments)
        
       
#         segmented_signals = torch.stack(segmented_signals) 

#         return segmented_signals, torch.tensor(condition, dtype=torch.float32)
import torch
from torch.utils.data import Dataset

import torch
from torch.utils.data import Dataset
class GWSignalDataset(Dataset):
    def __init__(
        self, 
        signals, 
        pro_signals, 
        pro_masked_signals, 
        conditions
    ):
        # 强制所有输入转为张量 -------------------------------------------------
        # 使用辅助方法 _ensure_tensor 处理转换逻辑
        self.signals = self._ensure_tensor(signals)
        self.pro_signals = self._ensure_tensor(pro_signals)
        self.pro_masked_signals = self._ensure_tensor(pro_masked_signals)
        self.conditions = self._ensure_tensor(conditions)
        
        # 统一设备到CPU（避免后续DataLoader潜在问题）
        self._to_cpu()
        
    def _ensure_tensor(self, data) -> torch.Tensor:
        """将输入转换为张量，支持列表、NumPy数组或已存在的张量"""
        if not isinstance(data, torch.Tensor):
            try:
                # 转换时保留原始数据类型（若需调整类型，可在此指定 dtype）
                tensor = torch.as_tensor(data)
                return tensor
            except Exception as e:
                raise ValueError(f"数据无法转换为张量: {type(data)} -> {e}")
        return data
    
    def _to_cpu(self):
        """确保所有张量位于CPU（避免混用GPU张量）"""
        self.signals = self.signals.cpu()
        self.pro_signals = self.pro_signals.cpu()
        self.pro_masked_signals = self.pro_masked_signals.cpu()
        self.conditions = self.conditions.cpu()
        
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        return (
            self.signals[idx],
            self.pro_signals[idx],
            self.pro_masked_signals[idx],
            self.conditions[idx]
        )
class GWSignalWithNoiseDataset(Dataset):
    def __init__(self, signals,pro_signals,pro_masked_signals,pro_masked_datas, conditions):
        """
        :param signals: 经过小波变换和分段处理的原始信号
        :param masked_signals: 经过小波变换和分段处理的掩码信号
        :param gap_start: 掩码的起始位置
        :param conditions: 额外的条件信息
        """
        self.signals = signals
        self.pro_signals=pro_signals
        self.pro_masked_signals = pro_masked_signals
        self.pro_masked_datas=pro_masked_datas
        self.conditions = conditions

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        pro_signal = self.pro_signals[idx]
        pro_masked_signal = self.pro_masked_signals[idx]
        pro_masked_datas = self.pro_masked_datas[idx]
        condition = self.conditions[idx]
    

        return (
            torch.tensor(signal, dtype=torch.float32),
            torch.tensor(pro_signal, dtype=torch.float32),
            torch.tensor(pro_masked_signal, dtype=torch.float32),
            torch.tensor(pro_masked_datas, dtype=torch.float32),
            torch.tensor(condition, dtype=torch.float32)
        )
class GWSignalDatasetForDecomposition(Dataset):
    def __init__(self, signals,pro_signals,signals_22,signals_21,signals_33,signals_44, conditions):
        """
        :param signals: 经过小波变换和分段处理的原始信号
        :param signals_22:  22 模信号
        :param signals_21:  21 模信号
        :param signals_33:  33 模信号
        :param signals_44:  44 模信号
        :param conditions: 额外的条件信息
        """
        self.signals = signals
        self.pro_signals=pro_signals
        self.signals_22 = signals_22
        self.signals_21 = signals_21
        self.signals_33 = signals_33
        self.signals_44 = signals_44
        self.conditions = conditions

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        pro_signal = self.pro_signals[idx]
        signal_22 = self.signals_22[idx]
        signal_21 = self.signals_21[idx]
        signal_33 = self.signals_33[idx]
        signal_44 = self.signals_44[idx]
        condition = self.conditions[idx]

        return (
            torch.tensor(signal, dtype=torch.float32),
            torch.tensor(pro_signal, dtype=torch.float32),
            torch.tensor(signal_22, dtype=torch.float32),
            torch.tensor(signal_21, dtype=torch.float32),
            torch.tensor(signal_33, dtype=torch.float32),
            torch.tensor(signal_44, dtype=torch.float32),
            #torch.tensor(gap_start, dtype=torch.long),
            torch.tensor(condition, dtype=torch.float32)
        )