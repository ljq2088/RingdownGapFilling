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

    def forward(self, src, key_padding_mask=None):
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
            # attn_output, _ = self.self_attn(src_ch_transposed, src_ch_transposed, src_ch_transposed)
            
                # broadcast to match attention score shape: (num_tokens, batch_size, num_tokens)
            
            #attn_output, _ = self.self_attn(src_ch_transposed, src_ch_transposed, src_ch_transposed,key_padding_mask=key_padding_mask)
            # src_ch_transposed: [T, B, D] -> 转换成 [B, T, D]
            src_ch_seq = src_ch_transposed.transpose(0, 1)  # [B, T, D]

            # 1️⃣ Q, K, V 直接取为原始输入（你也可以添加 Linear 层投影）
            Q = src_ch_seq  # [B, T, D]
            K = src_ch_seq
            V = src_ch_seq
            d_k = embedding_dim ** 0.5

            # 2️⃣ 计算注意力分数 [B, T, T]
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k  # [B, T, T]

            # 3️⃣ 添加 soft bias（你之前的 attn_bias）
            if key_padding_mask is not None:
                # key_padding_mask: [B, T] -> [B, 1, T]
                soft_bias = key_padding_mask.unsqueeze(1).to(attn_scores.dtype)  # float32
                attn_scores = attn_scores - soft_bias * 5.0  # 可调节抑制强度

            # 4️⃣ 归一化注意力
            attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, T, T]

            # 5️⃣ 加权求值
            attn_output = torch.matmul(attn_weights, V)  # [B, T, D]

            # 6️⃣ 转回 PyTorch 多头 attention 的格式 [T, B, D]
            attn_output = attn_output.transpose(0, 1)  # [T, B, D]

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

    def forward(self, x, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x