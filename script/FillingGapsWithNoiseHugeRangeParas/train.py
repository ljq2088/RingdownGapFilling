import torch 
import sys
sys.path.append('/home/ljq/code/RingdownGapFilling')
from dataset.dataset import *
# 加载保存的 Dataset
dataset = torch.load('/home/ljq/code/RingdownGapFilling/dataset/train_dataset_signal_data_noise_gaps_HugeRangeParas.pth',weights_only=False)
Spec_dataset=torch.load('/home/ljq/code/RingdownGapFilling/dataset/train_dataset_signal_data_noise_gaps_HugeRangeParas_Spec.pth',weights_only=False)

dataset= CombinedDataset(dataset, Spec_dataset)


import torch
import torch.nn 
from torch.utils.data import DataLoader
from config.config import Config
from data.data_generation import generate_data
from dataset.dataset import GWSignalDataset
#from model.mymodel import MaskedConditionalAutoencoder
#from model.mymodel_2 import MaskedConditionalGapFiller

from model.DGFmodel import *
from trainer.trainer_pre_finetune import train_the_model
from utils.visualization import visualize_waveform
from utils.normalize import normalize
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.mask import generate_continuous_mask
from utils.wavelet import wavelet_bandpass
from utils.segment import segment_signal
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#dataset=new_dataset
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])




batch_size = Config.batch_size


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 获取一个批次的数据
targets,mask_1d,inputs, conditions,_ = next(iter(train_loader))

#打印形状
# print(f'Inputs shape: {inputs.shape}')  
# print(f'Targets shape: {targets.shape}')  
# print(f'Conditions shape: {conditions.shape}')  

import matplotlib.pyplot as plt

# 可视化第一个批次中的前几个样本
# num_samples_to_plot = 3  # 要绘制的样本数量
# for i in range(num_samples_to_plot):
#     plt.figure(figsize=(10, 4))
#     plt.plot(inputs[i].cpu().numpy(), label='Input (Masked)')
#     plt.plot(targets[i].cpu().numpy(), label='Target (Original)')
#     plt.title(f'Sample {i+1}')
#     plt.xlabel('Time Step')
#     plt.ylabel('Signal Value')
#     plt.legend()

#     plt.show()

print("targets.shape:",targets.shape)
print("inputs.shape:",inputs.shape)
print("mask_1d.shape:",mask_1d.shape)

SAVE_PATH = '/home/ljq/code/RingdownGapFilling/saved_models/noise/lowSNR/model_with_noise_gaps_HugeRangeParas'
# 定义模型  
model = DGF().to(device)
train_the_model(
    model,
    train_loader,
    val_loader,
    num_epochs=Config.num_epochs,
    learning_rate=Config.learning_rate,
    save_path= SAVE_PATH,
    device=device,
    save_freq=10
)
