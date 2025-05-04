import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config.config import Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ConditionMLP(nn.Module):
    def __init__(self, input_dim, output_dim=Config.EMBEDDING_dim):
        super(ConditionMLP, self).__init__()
        # 定义 MLP：输入维度为 input_dim，输出维度为 output_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, Config.segment_length),  # 第一个线性层，映射到128维
            nn.ReLU(),                  # 激活函数
            nn.Linear(Config.segment_length, output_dim)   # 映射到目标的 64 维
        )

    def forward(self, x):
        # 前向传播
        return self.mlp(x)



class ConditionConv1D(nn.Module):
    def __init__(self, input_channels=1, output_channels=Config.num_token):
        super(ConditionConv1D, self).__init__()
        # 定义 1D 卷积层
        # in_channels = 1 表示输入通道数为 1，out_channels = 32 表示输出通道数为 32
        # kernel_size = 3 表示卷积核大小，padding = 1 确保输入输出的长度保持不变
        self.conv1d = nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 输入 x 形状为 (batch_size, 64)
        # 先扩展成 (batch_size, 1, 64) 以适应 Conv1d 输入格式
        x = x.unsqueeze(1)

        # 通过 1D 卷积，将形状变为 (batch_size, 32, 64)
        x = self.conv1d(x)

        # 激活函数
        x = self.relu(x)

        return x





class ConditionEmbedding(nn.Module):
    def __init__(self):
        super(ConditionEmbedding, self).__init__()
        # 定义 1D 卷积层
        self.mlp = ConditionConv1D()
        # 定义 2D 卷积层
        self.conv1d = ConditionConv1D()
        self.conv2dembedding = Conv2DEMbedding()
    def forward(self, x):
        x = self.mlp(x)

        x = self.conv1d(x)
        x = self.conv2dembedding(x)

        return x    