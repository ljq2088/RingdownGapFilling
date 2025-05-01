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
# def estimate_physical_parameters(signal):
#     """
#     估计信号的物理参数，例如频率、相位和振幅。
#     这是一个简单的示例实现，具体实现应根据你的需求进行调整。
#     """
#     # 假设信号是一个简单的正弦波，我们可以估计它的幅度、频率和相位
#     amplitude = signal.abs().max().item()  # 估计振幅
#     frequency = torch.fft.fftfreq(signal.size(0)).abs().argmax().item()  # 估计频率
#     phase = torch.atan2(signal[1], signal[0]).item()  # 估计相位

#     return [amplitude, frequency, phase]
 



 

# 生成训练集数据
#freq_ifft = np.arange(Config.f_in,Config.fout,Config.f_step)

signal_length=Config.signal_length
gap_size = signal_length//Config.signal_to_gap_length_ratio


data = np.load('data/signal_data.npz')


# 查看 npz 文件中包含的键
print("Keys in .npz file:", data.keys())

# 假设信号存储在键 'signals' 下
if 'signals' in data:
    signals = data['signals']  # 加载信号
    print(f"Signals array shape: {signals.shape}")
    
    incorrect_count = 0  # 用于计数不符合条件的信号
    max_reports = 10  # 最大报告数量，防止输出过多
    
    # 检查每个信号的 dtype 和 shape
    for i, signal in enumerate(signals):
        if signal.dtype != np.float32 or signal.shape != (1056,):
            incorrect_count += 1
            if incorrect_count <= max_reports:
                print(f"Signal {i} dtype: {signal.dtype}")
                print(f"Signal {i} shape: {signal.shape}")
            
            # 如果已经打印足够多的报告，提前终止循环
            if incorrect_count == max_reports:
                print("More incorrect signals exist, but only the first 10 are displayed.")
                break
    
    if incorrect_count == 0:
        print("All signals have the correct dtype and shape.")
else:
    print("No 'signals' key found in the .npz file")


#data = np.load('data/test_signal_data.npz')
signals = data['signals']
conditions = data['conditions']
#signal,condition=generate_data(1)
# signals=[]
# signal=signal.transpose(0,1)
# print(signal.shape)
# for _ in range(Config.num_samples):
    
#     signals.append(signal)

# #masked_signals = data['masked_signals']

# conditions=[]
# print(condition.shape)
# condition=condition.transpose(1,0)
# for _ in range(Config.num_samples):
    
#     conditions.append(condition)


#数据预处理
#n_signals,mean,std=normalize(signals)
signals = torch.tensor(signals, dtype=torch.float32)
masks = generate_continuous_mask(signals.size(0), signal_length, gap_size)
masked_signals = []
signals_copy=signals
signals,_,_=normalize(signals)
for i in range(signals_copy.size(0)):
    masked_signal = np.copy(signals_copy[i])
    masked_signal[~masks[i].numpy()] = 0  # 将掩码位置的信号置为0
    masked_signals.append(masked_signal)
masked_signals,_,_=normalize(masked_signals)
processed_signals = []
processed_masked_signals = []

i=0
for signal, masked_signal in zip(signals, masked_signals):
    # 对原始信号进行小波变换和分段
    transformed_signal = wavelet_bandpass(signal)
    segmented_signal = segment_signal_IMR(transformed_signal)

    # 对掩码信号进行小波变换和分段
    transformed_masked_signal = wavelet_bandpass(masked_signal)
    segmented_masked_signal = segment_signal_IMR(transformed_masked_signal)

    processed_signals.append(segmented_signal)
    processed_masked_signals.append(segmented_masked_signal)
    if i%100==0:
        print(i)    
    i+=1
    
processed_signals = torch.tensor(processed_signals, dtype=torch.float32)
processed_masked_signals = torch.tensor(processed_masked_signals, dtype=torch.float32)
conditions = torch.tensor(conditions, dtype=torch.float32)

dataset = GWSignalDataset(signals,processed_signals,processed_masked_signals, conditions)






import os
import torch
from torch.utils.data import DataLoader

# 假设 dataset 是您已经准备好的 GWSignalDataset
#dataset = GWSignalDataset(signals, processed_signals, processed_masked_signals, conditions)

batch_size_save = 5000  # 每批次保存的数据量
output_dir = 'data/gap'

# 确保目录存在
os.makedirs(output_dir, exist_ok=True)

# 分批保存数据
for i in range(0, len(dataset), batch_size_save):
    batch_data = []
    
    for j in range(i, min(i + batch_size_save, len(dataset))):
        signal, pro_signal, pro_masked_signal, condition = dataset[j]
        batch_data.append({
            'signal': signal,
            'pro_signal': pro_signal,
            'pro_masked_signal': pro_masked_signal,
            'condition': condition
        })
    
    # 保存当前批次数据
    file_path = os.path.join(output_dir, f'batch_{i // batch_size_save}.pt')
    torch.save(batch_data, file_path)  # 保存当前批次的数据
    print(f'Saved batch {i // batch_size_save} to {file_path}')
import torch
import os
from dataset.dataset import GWSignalDataset  # 假设GWSignalDataset已经定义好

output_dir = 'data/gap'  # 保存批次数据的目录
all_signals = []
all_pro_signals = []
all_pro_masked_signals = []
all_conditions = []

# 加载并合并所有批次的数据
for file_name in os.listdir(output_dir):
    if file_name.endswith('.pt'):
        file_path = os.path.join(output_dir, file_name)
        batch_data = torch.load(file_path)  # 加载当前批次的数据
        
        for data in batch_data:
            all_signals.append(data['signal'])
            all_pro_signals.append(data['pro_signal'])
            all_pro_masked_signals.append(data['pro_masked_signal'])
            all_conditions.append(data['condition'])

# 将所有数据合并成一个大的 Tensor
all_signals_tensor = torch.stack(all_signals, dim=0)
all_pro_signals_tensor = torch.stack(all_pro_signals, dim=0)
all_pro_masked_signals_tensor = torch.stack(all_pro_masked_signals, dim=0)
all_conditions_tensor = torch.stack(all_conditions, dim=0)

# 创建一个新的 GWSignalDataset 实例
new_dataset = GWSignalDataset(
    signals=all_signals_tensor,
    pro_signals=all_pro_signals_tensor,
    pro_masked_signals=all_pro_masked_signals_tensor,
    conditions=all_conditions_tensor
)

# 查看合并后的数据集大小
print(f"合并后的数据集大小: {len(new_dataset)}")
