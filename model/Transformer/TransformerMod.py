import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config.config import Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TransformerEncoderLayerWithChannels(nn.Module):
    def __init__(self, embedding_dim=Config.EMBEDDING_dim, num_heads=Config.num_heads, dim_feedforward=Config.FF_dim, dropout=Config.dropout):
        super(TransformerEncoderLayerWithChannels, self).__init__()
        
        # Multi-head Self-Attention
        self.self_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout)
        
        # Feedforward Network
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, embedding_dim),
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src 的形状为 (batch_size, channels, num_tokens, embedding_dim)
        batch_size, channels, num_tokens, embedding_dim = src.shape
        
        # 我们对每个 channel 独立应用 self-attention
        outputs = []
        for ch in range(channels):
            # 对于每个 channel，进行 multi-head self-attention
            src_ch = src[:, ch, :, :]  # 取出当前 channel 的数据，形状为 (batch_size, num_tokens, embedding_dim)
            
            # Self-Attention expects input as (num_tokens, batch_size, embedding_dim)
            src_ch_transposed = src_ch.transpose(0, 1)  # 转置为 (num_tokens, batch_size, embedding_dim)
            
            # Self-Attention, output shape: (num_tokens, batch_size, embedding_dim)
            attn_output, _ = self.self_attn(src_ch_transposed, src_ch_transposed, src_ch_transposed)
            
            # Residual Connection + Layer Normalization
            src2 = src_ch_transposed + self.dropout(attn_output)
            src2 = self.norm1(src2)
            
            # Feedforward Layer
            src2_transposed = src2.transpose(0, 1)  # 转回 (batch_size, num_tokens, embedding_dim)
            feedforward_output = self.feedforward(src2_transposed)
            
            # Residual Connection + Layer Normalization
            src3 = src2_transposed + self.dropout(feedforward_output)
            output = self.norm2(src3)
            
            # 将处理好的 channel 加入 outputs 列表
            outputs.append(output.unsqueeze(1))  # (batch_size, 1, num_tokens, embedding_dim)
        
        # 拼接所有 channels
        outputs = torch.cat(outputs, dim=1)  # 最终形状为 (batch_size, channels, num_tokens, embedding_dim)
        
        return outputs

class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers=Config.num_layers_T, dim_feedforward=Config.FF_dim, dropout=Config.dropout):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayerWithChannels(embedding_dim, num_heads, dim_feedforward, dropout) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x