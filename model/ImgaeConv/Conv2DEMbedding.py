import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config.config import Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Conv2DEMbedding(nn.Module):
    def __init__(self, in_channels=1, out_channels=Config.channels, token_dim=Config.segment_length, embedding_dim=Config.EMBEDDING_dim, kernel_size=Config.ConEkernel_size, padding=Config.ConEpadding):
        super(Conv2DEMbedding, self).__init__()
        # 定义 2D 卷积层
        # Conv2d: in_channels=1, out_channels=8, kernel_size=3x3, padding=1 保持输入的空间维度不变
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(token_dim, embedding_dim)

    def forward(self, x):
        # 输入 x 的形状为 (batch_size, 32, 64)
        # 先通过 unsqueeze 添加维度，变为 (batch_size, 1, 32, 64)
        x = x.unsqueeze(1)
        
        # 通过 2D 卷积，变为 (batch_size, 8, 32, 64)
        x = self.conv2d(x)

        # 激活函数
        x = self.relu(x)
        x=self.linear(x)

        return x