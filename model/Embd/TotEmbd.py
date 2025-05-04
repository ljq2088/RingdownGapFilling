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

    def forward(self, x,qt):
        # 计算 Token Embedding, Convolutional Embedding, Position Embedding
        # print(f"input shape: {x.shape}")
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
        
        return RF