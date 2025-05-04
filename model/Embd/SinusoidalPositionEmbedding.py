import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config.config import Config
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, num_tokens, token_dim, output_dim=Config.EMBEDDING_dim):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.embedding_dim = token_dim
        
        # 创建正弦位置嵌入矩阵
        position = torch.arange(0, num_tokens, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, token_dim, 2).float() * (-math.log(10000.0) / token_dim))
        
        # 正弦和余弦函数的嵌入
        pe = torch.zeros(num_tokens, token_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 将正弦位置嵌入注册为常量
        self.register_buffer('pe', pe)
        
        # 线性映射层，将正弦位置嵌入映射到 output_dim
        self.linear = nn.Linear(token_dim, output_dim)

    def forward(self, x):
        # x 的形状为 (batch_size, channels, num_tokens, embedding_dim)
        batch_size, channels, num_tokens, embedding_dim = x.shape
        
        # 扩展位置嵌入的维度以适应输入
        position_embeds = self.pe[:num_tokens, :].unsqueeze(0).unsqueeze(0)  # 形状 (1, 1, num_tokens, embedding_dim)
        position_embeds = position_embeds.expand(batch_size, channels, -1, -1)  # 形状 (batch_size, channels, num_tokens, embedding_dim)
        
        # 线性映射位置嵌入
        position_embeds = self.linear(position_embeds)  # 形状变为 (batch_size, channels, num_tokens, output_dim)
        
        return position_embeds

