import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config.config import Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TokenEmbedding(nn.Module):
    def __init__(self, token_dim, embedding_dim=Config.EMBEDDING_dim):
        super(TokenEmbedding, self).__init__()
        # 定义线性层，将 token_dim 映射到 embedding_dim
        self.linear = nn.Linear(token_dim, embedding_dim)
    
    def forward(self, x):
        # x 的形状为 (batch_size, channels, num_tokens, token_dim)
        # 我们对最后一维 (token_dim) 进行线性映射
        x = self.linear(x)
        # 返回形状为 (batch_size, channels, num_tokens, embedding_dim)
        return x