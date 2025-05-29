from model.Embd.TokenEmbedding import TokenEmbedding
from model.Embd.SinusoidalPositionEmbedding import SinusoidalPositionEmbedding
from model.Embd.ConvEmbeddingWithLinear import ConvEmbeddingWithLinear
from model.Embd.CondConvEmbd import ConditionEmbedding
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config.config import Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from config.config import Config
class TotalEmbedding(nn.Module):
    def __init__(self, token_dim, embedding_dim, conv_channels, conv_out_channels, num_tokens, dropout_p=0.1):
        super(TotalEmbedding, self).__init__()
        # 初始化各个嵌入模块
        self.register_parameter(
    "mask_patch", nn.Parameter(torch.randn(1, Config.channels, 1, token_dim))  # [B, C, T, token_len]
)

        self.token_embedding = TokenEmbedding(token_dim, embedding_dim)
        self.conv_embedding = ConvEmbeddingWithLinear(conv_channels, conv_out_channels, embedding_dim=embedding_dim)
        self.position_embedding = SinusoidalPositionEmbedding(num_tokens, embedding_dim)
        #self.condition_embedding=ConditionEmbedding()
        # 2D 卷积层，用于之后的卷积操作
        self.conv2d = nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=(3, 3), padding=1)
        
        # 激活函数 GeLU
        self.gelu = nn.GELU()

        # Dropout
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x,qt,soft_mask=None, mask_token=None):
        # x: 输入信号，形状为 (batch_size, channels, num_tokens, token_dim)
        # qt: Q变换后且经过TimeMixer的信号，形状为 (batch_size, channels, num_tokens, token_dim)
        # soft_mask: 可选的 soft mask，形状为 (batch_size, num_tokens)
        # mask_token: 可选的 mask token，形状为 (1, 1, embedding_dim)
        # 计算 Token Embedding, Convolutional Embedding, Position Embedding
        # print(f"input shape: {x.shape}")
        # 替换输入 token 为 mask_patch（方案A）
        if soft_mask is not None:
            # 构造 token-level hard mask：[B, 32]
            #token_hard_mask = (soft_mask > 0.5).float()  # 0/1

            # reshape: [B, 1, 32, 1] → 方便 broadcast 到 [B, C=8, 32, 64]
            soft_mask = soft_mask.unsqueeze(1).unsqueeze(-1)
            # print(f"soft_mask shape: {soft_mask.shape}")
            # print(f"x shape: {x.shape}")
            # print(f"mask_patch shape: {self.mask_patch.shape}")
            # 替换被mask的 token 段（将 x 中的对应 token patch 替换为 mask_patch）
            x = x * (1 - soft_mask) + self.mask_patch * soft_mask
         
        # print(f"qt shape: {qt.shape}")
        x=x+qt
        token_embed = self.token_embedding(x)
        conv_embed = self.conv_embedding(x)
        position_embed = self.position_embedding(x)
        #condition_embed=self.condition_embedding(x)
       

        # DF = TE + CE + PE
        DF = token_embed + conv_embed + position_embed#+condition_embed
        
        # 进行 2D 卷积并加上残差连接
        conv_out = self.conv2d(DF)  # 卷积操作
        conv_out = self.gelu(conv_out)  # GeLU 激活
        DF_with_conv = DF + conv_out  # 残差项
        
        # Dropout 操作
        RF = self.dropout(DF_with_conv)
        if soft_mask is not None and mask_token is not None:
            # soft_mask: [B, 32] → reshape 成 [B, 1, 32, 1] 以与 RF [B, 8, 32, D] 广播
            # soft_mask = soft_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, 32, 1]
            mask_token_expanded = mask_token.view(1, 1, 1, -1)  # [1, 1, 1, D]
            # print(f"RF shape: {RF.shape}")
            # print(f"mask_token shape: {mask_token_expanded.shape}")
            # print(f"soft_mask shape: {soft_mask.shape}")
            RF = RF * (1 - soft_mask) + mask_token_expanded * soft_mask  # broadcast
        return RF