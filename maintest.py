import os
import numpy as np
from tqdm import tqdm

# 临时文件夹路径
TEMP_DIR = "data/temp_files"

# 合并文件路径
OUTPUT_FILE = "data/merged_signal_data.npz"

def combine_data(temp_dir, output_file):
    signals = []
    conditions = []

    # 遍历文件夹中的所有 npz 文件
    temp_files = [f for f in os.listdir(temp_dir) if f.endswith('.npz')]
    temp_files.sort()  # 如果需要按顺序合并，可以先排序

    print(f"Found {len(temp_files)} files to combine. Starting...")

    # 加载每个 npz 文件
    for file_name in tqdm(temp_files, desc="Combining files"):
        file_path = os.path.join(temp_dir, file_name)
        with np.load(file_path) as data:
            signals.append(data['signal'])
            conditions.append(data['condition'])

    # 堆叠所有数据
    signals = np.stack(signals)
    conditions = np.stack(conditions)

    # 保存到输出文件
    np.savez(output_file, signals=signals, conditions=conditions)
    print(f"Combined data saved to {output_file}")

# 调用函数
combine_data(TEMP_DIR, OUTPUT_FILE)
import os
import hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from config.config import Config
from data.data_generation import generate_data
from dataset.dataset import GWSignalDataset
#from model.mymodel import MaskedConditionalAutoencoder
from model.mymodel_2 import MaskedConditionalGapFiller
from trainer.trainer import train_the_model
from utils.visualization import visualize_waveform
from utils.normalize import normalize
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.mask import generate_continuous_mask
from utils.wavelet import wavelet_bandpass
from utils.segment import segment_signal_IMR
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_cache_filename(original_file):
    # 生成原始文件内容的哈希值作为缓存文件名
    with open(original_file, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return f"preprocessed_cache_{file_hash}.npz"
def load_or_preprocess_data(original_file, signal_length, gap_size):
    cache_file = get_cache_filename(original_file)
    
    if os.path.exists(cache_file):
        print("Loading cached preprocessed data...")
        data = np.load(cache_file)
        signals = data['signals']
        conditions = data['conditions']
        masks = data['masks']
        processed_signals = data['processed_signals']
        processed_masked_signals = data['processed_masked_signals']
        norm_mean = data['norm_mean']
        norm_std = data['norm_std']
        return signals, conditions, masks, processed_signals, processed_masked_signals, norm_mean, norm_std
    else:
        print("Preprocessing data and saving cache...")
        data = np.load(original_file)
        signals = data['signals']
        conditions = data['conditions']
        
        # 生成掩码
        signals_tensor = torch.tensor(signals, dtype=torch.float32)
        masks = generate_continuous_mask(signals_tensor.size(0), signal_length, gap_size)
        
        # 归一化原始信号
        normalized_signals, norm_mean, norm_std = normalize(signals_tensor)
        
        # 处理掩码信号
        masked_signals = []
        for i in range(signals_tensor.size(0)):
            masked_signal = signals_tensor[i].clone().numpy()  # 转换为NumPy数组
            masked_signal[~masks[i].numpy()] = 0  # 将掩码位置的信号置为0
            masked_signals.append(masked_signal)
        
        # 调试：检查 masked_signals 的形状和内容
        print(f"masked_signals length: {len(masked_signals)}")
        print(f"First masked_signal shape: {masked_signals[0].shape}")
        print(f"First masked_signal content: {masked_signals[0]}")
        
        # 归一化掩码信号
        masked_signals, _, _ = normalize(masked_signals)
        
        # 小波变换和分段处理
        processed_signals = []
        processed_masked_signals = []
        for signal, masked_signal in zip(normalized_signals, masked_signals):
            transformed_signal = wavelet_bandpass(signal)
            segmented_signal = segment_signal_IMR(transformed_signal)
            processed_signals.append(segmented_signal)
            
            transformed_masked_signal = wavelet_bandpass(masked_signal)
            segmented_masked_signal = segment_signal_IMR(transformed_masked_signal)
            processed_masked_signals.append(segmented_masked_signal)
        
        # 保存到缓存文件
        np.savez(
            cache_file,
            signals=normalized_signals.numpy(),
            conditions=conditions,
            masks=masks.numpy(),
            processed_signals=np.array(processed_signals),
            processed_masked_signals=np.array(processed_masked_signals),
            norm_mean=[norm_mean],
            norm_std=[norm_std]
        )
        return normalized_signals, conditions, masks, processed_signals, processed_masked_signals, norm_mean, norm_std
original_data_file = 'data/merged_signal_data.npz'
signal_length = Config.signal_length
gap_size = signal_length // Config.signal_to_gap_length_ratio

# 使用缓存加载或生成预处理数据
signals, conditions, masks, processed_signals, processed_masked_signals, norm_mean, norm_std = load_or_preprocess_data(
    original_data_file, signal_length, gap_size
)

# 将数据转换为PyTorch张量（已归一化）
dataset = GWSignalDataset(
    signals=torch.tensor(signals, dtype=torch.float32),
    pro_signals=torch.tensor(processed_signals, dtype=torch.float32),
    pro_masked_signals=torch.tensor(processed_masked_signals, dtype=torch.float32),
    conditions=torch.tensor(conditions, dtype=torch.float32)
)
