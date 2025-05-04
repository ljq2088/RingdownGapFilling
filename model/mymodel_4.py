import torch
import torch.nn as nn
from config.config import Config
import math
from model.QTranTimeMixerMod import *
from model.Embd.TokenEmbedding import TokenEmbedding
from model.Embd.SinusoidalPositionEmbedding import SinusoidalPositionEmbedding
from model.Embd.ConvEmbeddingWithLinear import ConvEmbeddingWithLinear
from model.ImgaeConv.Conv2DEMbedding import Conv2DEMbedding
from model.Embd.CondConvEmbd import ConditionEmbedding
from model.Transformer.TransformerMod import *
from model.MLP.MLP import MLP
from model.Embd.TotEmbd import TotalEmbedding
from model.CombSeg.Combine_segments import combine_segments
from model.QTranTimeMixerMod import QTransformModule
#Embedding
import torch
import math
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')





class Encoder(nn.Module):
    def __init__(self,channels=Config.channels, token_dim=Config.segment_length, embedding_dim=Config.EMBEDDING_dim, num_heads=Config.num_heads, num_layers=Config.num_layers_T, dropout=Config.dropout):
        super(Encoder, self).__init__()
        #self.token_embedding = TokenEmbedding(token_dim, embedding_dim)
        #self.conv_embedding = ConvEmbeddingWithLinear(channels, token_dim, embedding_dim=embedding_dim)
        #self.position_embedding = SinusoidalPositionEmbedding(Config.num_token, token_dim, embedding_dim)
        self.totalembedding=TotalEmbedding(token_dim, embedding_dim, channels, channels, Config.num_token, Config.dropout)
        self.transformer = TransformerEncoder(embedding_dim=embedding_dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        self.QtranTimeMixer =TimeMixerEncoder(signal_length=Config.signal_length,num_token=Config.num_token,token_dim=Config.segment_length)  # Q变换模块

    def forward(self, x,qt):
        # 进行 Token Embedding, Convolutional Embedding 和 Position Embedding
        QtTimeMixer=self.QtranTimeMixer(qt)
        
        RF=self.totalembedding(x,QtTimeMixer)
       # print(f"RF shape: {RF.shape}")
       # print(f"QtTimeMixer shape: {QtTimeMixer.shape}")
        # Transformer Encoder
        output = self.transformer(RF)  # 形状为 (batch_size, num_tokens, embedding_dim)
        
        return output

class InverseTokenEmbedding(nn.Module):
    def __init__(self, token_embedding_layer):
        super(InverseTokenEmbedding, self).__init__()
        # 获取 TokenEmbedding 的权重并转置
        weight = token_embedding_layer.linear.weight
        self.inverse_linear = nn.Linear(weight.size(1), weight.size(0), bias=False)  # 定义逆映射层
        self.inverse_linear.weight = nn.Parameter(weight.T)  # 使用转置后的权重

    def forward(self, x):
        return self.inverse_linear(x)

class ChannelMerger(nn.Module):
    def __init__(self, input_channels=Config.channels, output_channels=1, signal_length=Config.signal_length):
        super(ChannelMerger, self).__init__()
        # 定义 Conv1d 卷积层，将输入 8 个通道的信号变换为 1 个输出信号
        # kernel_size=3, padding=1 用于保持输入输出信号长度不变
        self.conv1d = nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # 输入的 x 形状为 (batch_size, input_channels=8, signal_length=1056)
        # 通过 1D 卷积，将每个通道卷积成长度相同的信号
        output = self.conv1d(x)
        # 卷积的输出形状为 (batch_size, output_channels=1, signal_length=1056)
        return output  # 将 (batch_size, 1, 1056) 变为 (batch_size, 1056)

from ssqueezepy import icwt, Wavelet
import torch
import numpy as np




import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, inverse_token_embedding):
        super(Decoder, self).__init__()
        self.mlp = MLP()  # MLP模块
        self.inverse_token_embedding = inverse_token_embedding  # 逆 Token Embedding 模块
        self.CM=ChannelMerger()
    def forward(self, encoder_output):
        """
        :param encoder_output: 从 Encoder 得到的输出, 形状为 (batch_size, channels, num_tokens, embedding_dim)
                               即 (16, 8, 32, 128)
        :return: 最终重建的信号, 形状为 (batch_size, 1, signal_length) 即 (16, 1, 1056)
        """
        # Step 1: MLP 处理
        x = self.mlp(encoder_output)  # MLP 输出形状为 (16, 8, 32, 128)

        # Step 2: 通过逆 Token Embedding 将数据从 (16, 8, 32, 128) 转换为 (16, 8, 32, 64)
        x = self.inverse_token_embedding(x)  # 逆 Token Embedding 输出形状为 (16, 8, 32, 64)

        # Step 3: Segment 拼接，将分段信号重新组合成 (16, 8, 1056)
        x = combine_segments(x, segment_length=64, signal_length=1056, overlap=0.5)  # 拼接后形状为 (16, 8, 1056)

        # Step 4: 将 8 个通道的信号重建为 1 个通道
        x = self.CM(x)  # 小波逆变换后形状为 (16, 1, 1056)

        return x


class DenoiseMaskedGapsFiller(nn.Module):
    def __init__(self, channels=Config.channels, token_dim=Config.segment_length, embedding_dim=Config.EMBEDDING_dim, num_heads=Config.num_heads, num_layers=Config.num_layers_T, dropout=Config.dropout):
        super(DenoiseMaskedGapsFiller, self).__init__()
        self.encoder = Encoder(channels, token_dim, embedding_dim, num_heads, num_layers, dropout)
        token_embedding_instance = self.encoder.totalembedding.token_embedding
        self.inverse_token_embedding = InverseTokenEmbedding(token_embedding_instance)

        self.decoder = Decoder(self.inverse_token_embedding)
        
    
    def forward(self, x,qt):  
        #print(f"input qt shape: {qt.shape}")
        encoder_output = self.encoder(x,qt)
        output = self.decoder(encoder_output)
        return output










