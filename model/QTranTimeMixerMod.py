import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ssqueezepy import cwt  # 使用小波变换计算Q变换频谱图
from config.config import Config
down_samp =Config.down_samp  # 下采样因子
scale_f=Config.scale_f  # 小波变换的尺度数
# Q变换模块，用于将输入的时间序列转换为时频图
class QTransformModule(nn.Module):
    def __init__(self, wavelet="morlet", scales=scale_f):
        super(QTransformModule, self).__init__()
        self.wavelet = wavelet  # 小波选择
        self.scales = np.linspace(1, scales, num=scales)  # 将scales转为一个线性数组 # 小波变换的尺度数

    def forward(self, x):
        """
        输入, x 为时间序列数据，形状为 (batch_size, signal_length)
        输出：振幅谱和相位谱
        """
        # Q变换：计算CWT（连续小波变换），返回复数形式的时频表示
        # shape of x: (batch_size, signal_length)
        # 使用连续小波变换（CWT）来模拟Q变换
        cwt_output, _ = cwt(x.numpy(), self.wavelet, self.scales)  # 结果是复数形式的时频图
        cwt_output = torch.tensor(cwt_output, dtype=torch.complex64)  # 转换为PyTorch张量
        #对时间方向进行2次下采样
        #cwt_output = cwt_output[:, :, ::down_samp]
        cwt_output = cwt_output[:, ::down_samp]  # 下采样
        # 提取振幅谱和相位谱
        amplitude = torch.abs(cwt_output)  # 振幅谱
        phase = torch.angle(cwt_output)    # 相位谱
        
        #在dim=1开一个维度并拼接amplitude和phase
        output=torch.cat((amplitude.unsqueeze(0), phase.unsqueeze(0)), dim=0)  # 形状 (batch_size, 2, freq_bins, time_steps)
        return output
#应用：
# qtransform_module = QTransformModule()
# output = qtransform_module(x)

# 频谱嵌入模块，使用卷积和全连接层提取特征
class FrequencyEmbedding(nn.Module):
    def __init__(self, input_dim_f,input_dim_t, output_dim, channels=Config.channels):
        super(FrequencyEmbedding, self).__init__()
        self.channels=channels
        # 使用卷积层处理频谱图的输入
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1)  # 使用 stride=2 缩小特征图尺寸
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 使用 stride=2 缩小特征图尺寸
        
        self.conv3 = nn.Conv2d(32, self.channels, kernel_size=3, stride=2, padding=1)  # 使用 stride=2 缩小特征图尺寸
        
        # 使用一个 dummy 数据来计算卷积层输出的形状
        with torch.no_grad():
            dummy_input = torch.zeros(1, 2, input_dim_f ,input_dim_t)  
            self.flattened_size = self._get_conv_output(dummy_input)
        
        # 通过全连接层映射到目标空间
        self.fc1 = nn.Linear(self.flattened_size, output_dim)

    def _get_conv_output(self, shape):
        # 通过卷积层计算输出的形状
        x = self.conv1(shape)
        x = self.conv2(x)
        x = self.conv3(x)
        
        return x.view(x.size(0), -1).size(1)  # 展平后的维度

    def forward(self, x):
        """
        输入：振幅谱和相位谱, shape (batch_size, 2, freq_bins, time_steps)
        输出：频谱嵌入，适配后续深度学习模型
        """
        # 将振幅和相位合并为两个通道
        # x: (batch_size, channel,freq_bins, time_steps)
        
        #combined_input = torch.cat((x[:, 0, :, :].unsqueeze(1), x[:, 1, :, :].unsqueeze(1)), dim=1)  # 形状 (batch_size, 2, freq_bins, time_steps)
        #print(f"input shape: {x.shape}")  # 调试输出
        # 通过卷积层提取频域特征
        x = F.relu(self.conv1(x))  # 形状 (batch_size, 16, freq_bins/2, time_steps/2)
        x = F.relu(self.conv2(x))  # 形状 (batch_size, 32, freq_bins/4, time_steps/4)
        x = F.relu(self.conv3(x))  # 形状 (batch_size, 64, freq_bins/8, time_steps/8)
        #print(f"After conv layers shape: {x.shape}")  # 调试输出
        #将所有channel加起来成为(batch_size,freq_bins/8,time_steps/8)
        #x = torch.sum(x, dim=1)  # 形状 (batch_size, freq_bins/8, time_steps/8)
        #print(f"After sum shape: {x.shape}")  # 调试输出
        # 展平
        #print(f"Before flatten shape: {x.shape}")  # 调试输出
        x = x.view(x.size(0), -1)  # 展平为 (batch_size, flattened_size)
        #print(f"After flatten shape: {x.shape}")
        #print(f"After flatten shape: {x.shape}")
        # 通过全连接层映射到目标空间
        x = self.fc1(x)  # 形状 (batch_size, output_dim)
        return x


# 特征提取模块
class FeatureExtractionModule(nn.Module):
    def __init__(self, signal_length, num_token, token_dim):
        super(FeatureExtractionModule, self).__init__()
        #self.qtransform = QTransformModule()  # Q变换模块
        self.embedding = FrequencyEmbedding(scale_f,signal_length//down_samp, num_token * token_dim)  # 频谱嵌入模块
        self.num_token = num_token
        self.token_dim = token_dim
        
    def forward(self, x):
        """
        输入: x 为时间序列数据，形状为 (batch_size, signal_length)
        输出：嵌入特征，形状为 (batch_size, num_token, token_dim)
        """
        
        # 形状 (batch_size, 2, 256, 1056)
        # print(f"input shape: {x.shape}")
        x = self.embedding(x)  # 形状 (batch_size, num_token * token_dim)

        #重塑输出为 (batch_size, num_token, token_dim)
        x = x.view(-1, self.num_token, self.token_dim)  # 形状 (batch_size, num_token, token_dim)
        return x
class TimeMixerEncoder(nn.Module):
    def __init__(self, signal_length, num_token, token_dim, scale_f=scale_f,channels=Config.channels):  
        super(TimeMixerEncoder, self).__init__()
        #self.qtransform = QTransformModule()  # Q变换模块
        self.embedding = FrequencyEmbedding(scale_f, signal_length//down_samp, channels*num_token * token_dim)  # 频谱嵌入模块
        self.num_token = num_token
        self.token_dim = token_dim
        self.signal_length = signal_length
        self.channels=channels
        #MLP 层用于在通道维度混合
        self.mlp_channel = nn.Linear(2, 2)  # MLP 层用于在通道维度混合
        self.mlp_time= nn.Linear(self.signal_length//down_samp, self.signal_length//down_samp)  # MLP 层用于在时间维度混合
        self.mlp_feature= nn.Linear(scale_f ,scale_f)  # MLP 层用于在特征维度混合
        # MLP 层用于在特征维度和时间维度混合
        self.mlp_channel_2= nn.Linear(self.channels, self.channels)  # MLP 层用于在通道维度混合
        self.mlp_num_token = nn.Linear(num_token, num_token)
        self.mlp_token_dim= nn.Linear(token_dim, token_dim)

    def forward(self, x):
        """
        输入: x 为时间序列数据
        输出：嵌入特征，形状为 (batch_size, num_token, token_dim)
        """
        # Q变换
        #q_output = self.qtransform(x)  # (batch_size, 2, freq_bins, time_steps//down_samp)
        #混合
        
        x = self.mlp_time(x)  
        
        x=x.permute(0, 2, 3,1)  # 
        
        x = self.mlp_channel(x) 
        x = x.permute(0, 3, 2, 1)  # (batch_size, 2, time_steps//down_samp, freq_bins)
        
        x = self.mlp_feature(x)
        x = x.permute(0, 1, 3, 2)  # (batch_size, 2, freq_bins, time_steps//down_samp)
        # 特征嵌入
        x = self.embedding(x)  # (batch_size, channels*num_token * token_dim)
        x = x.view(-1,self.channels, self.num_token, self.token_dim)  # (batch_size, num_token, token_dim)
        #print(f"embedded shape: {x.shape}")
        x = self.mlp_token_dim(x)  #(batch_size, channels, num_token, token_dim)
        x=x.permute(0, 1,3,2)  # (batch_size, channels, token_dim, num_token)
        x = self.mlp_num_token(x) 
        x = x.permute(0,3,  2, 1)   #(batch_size, num_token, token_dim, channels)
        x = self.mlp_channel_2(x)
        x = x.permute(0, 3,1, 2)  # (batch_size, channels, num_token, token_dim)
       
        
      
        return x

#应用
# signal = torch.randn(16, 1056)  # 假设批量大小为16，信号长度为1056
# qt=QTransformModule()
# spec=qt(signal)
# print(f"spec shape: {spec.shape}")
# model = FeatureExtractionModule(signal_length=1056, num_token=32, token_dim=64)
# output = model(spec)
# print(f"Output shape: {output.shape}")
