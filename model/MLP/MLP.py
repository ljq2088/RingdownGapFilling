import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config.config import Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MLP(nn.Module):
    def __init__(self, input_dim=Config.EMBEDDING_dim//2, hidden_dim=Config.h_dim_MLP, output_dim=Config.EMBEDDING_dim):
        super(MLP, self).__init__()
        # 两层全连接层
        # print(input_dim)
        # print(f"input_dim: {input_dim}, type: {type(input_dim)}")
        # print(f"hidden_dim: {hidden_dim}, type: {type(hidden_dim)}")
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 第一层
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 第二层，将输出维度映射到 64

    def forward(self, x):
        # 形状: (batch_size, channels, num_tokens, embedding_dim)
        batch_size, channels, num_tokens, embedding_dim = x.shape
        #对最后一个维度下采样
        x = x[:, :, :, ::2]  # 下采样，保留每隔一个元素
        
        # 调整维度以适配全连接层
        x = x.view(batch_size * channels * num_tokens, embedding_dim//2)  # 展平成 (batch_size * channels * num_tokens, embedding_dim)
        #print(f"input shape: {x.shape}")
        x = self.fc1(x)  # 第一层全连接
        x = self.relu(x)  # 激活函数
        x = self.fc2(x)  # 第二层全连接，输出维度为 64

        # 恢复形状为 (batch_size, channels, num_tokens, output_dim)
        x = x.view(batch_size, channels, num_tokens, -1)
        return x
