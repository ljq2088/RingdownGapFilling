import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config.config import Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ConvEmbeddingWithLinear(nn.Module):
    def __init__(self, channels=Config.channels, conv_out_channels=Config.CEout_channels, kernel_size=Config.CEkernel_size, padding=Config.CEpadding, embedding_dim=Config.EMBEDDING_dim):
        super(ConvEmbeddingWithLinear, self).__init__()
        self.channels=channels
        # 定义 2D 卷积层
        self.conv = nn.Conv2d(in_channels=channels, out_channels=conv_out_channels, kernel_size=kernel_size, padding=padding)
        
        # 定义线性映射层，映射到最终的 embedding_dim
        self.linear = nn.Linear(Config.segment_length, embedding_dim)
        
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # x 的形状为 (batch_size, channels, num_tokens, token_dim)
        #print(f"x shape after convolution: {x.shape}")
        # Step 1: 进行 2D 卷积，保持 channels 不变
        conv_out = self.conv(x)  # 卷积输出的形状 (batch_size, channels, num_tokens, token_dim)
        conv_out = self.relu(conv_out)
        #print(f"conv_out shape after convolution: {conv_out.shape}")
        # Step 2: 线性层映射，仅映射 token_dim
        #batch_size, channels, num_tokens, token_dim = conv_out.shape
        
        # 将 token_dim 映射为 embedding_dim，形状为 (batch_size, channels, num_tokens, embedding_dim)
        #conv_out = conv_out.view(batch_size * channels, num_tokens, token_dim)
        embedding = self.linear(conv_out)  # 线性映射 token_dim -> embedding_dim
        #print(f"conv_out shape after convolution: {conv_out.shape}")
        # 将形状恢复为 (batch_size, channels, num_tokens, embedding_dim)
        #embedding = embedding.view(batch_size, self.channels, num_tokens, -1)
        
        return embedding
