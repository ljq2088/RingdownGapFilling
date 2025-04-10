import torch
import torch.nn as nn
from config.config import Config
import math


#Embedding
import torch
import math
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

class ConditionMLP(nn.Module):
    def __init__(self, input_dim, output_dim=Config.EMBEDDING_dim):
        super(ConditionMLP, self).__init__()
        # 定义 MLP：输入维度为 input_dim，输出维度为 output_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, Config.segment_length),  # 第一个线性层，映射到128维
            nn.ReLU(),                  # 激活函数
            nn.Linear(Config.segment_length, output_dim)   # 映射到目标的 64 维
        )

    def forward(self, x):
        # 前向传播
        return self.mlp(x)



class ConditionConv1D(nn.Module):
    def __init__(self, input_channels=1, output_channels=Config.num_token):
        super(ConditionConv1D, self).__init__()
        # 定义 1D 卷积层
        # in_channels = 1 表示输入通道数为 1，out_channels = 32 表示输出通道数为 32
        # kernel_size = 3 表示卷积核大小，padding = 1 确保输入输出的长度保持不变
        self.conv1d = nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 输入 x 形状为 (batch_size, 64)
        # 先扩展成 (batch_size, 1, 64) 以适应 Conv1d 输入格式
        x = x.unsqueeze(1)

        # 通过 1D 卷积，将形状变为 (batch_size, 32, 64)
        x = self.conv1d(x)

        # 激活函数
        x = self.relu(x)

        return x



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

class ConditionEmbedding(nn.Module):
    def __init__(self):
        super(ConditionEmbedding, self).__init__()
        # 定义 1D 卷积层
        self.mlp = ConditionConv1D()
        # 定义 2D 卷积层
        self.conv1d = ConditionConv1D()
        self.conv2dembedding = Conv2DEMbedding()
    def forward(self, x):
        x = self.mlp(x)

        x = self.conv1d(x)
        x = self.conv2dembedding(x)

        return x    
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

    def forward(self, x):
        # 计算 Token Embedding, Convolutional Embedding, Position Embedding
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

class Encoder(nn.Module):
    def __init__(self, channels=Config.channels, token_dim=Config.segment_length, embedding_dim=Config.EMBEDDING_dim, num_heads=Config.num_heads, num_layers=Config.num_layers_T, dropout=Config.dropout):
        super(Encoder, self).__init__()
        #self.token_embedding = TokenEmbedding(token_dim, embedding_dim)
        #self.conv_embedding = ConvEmbeddingWithLinear(channels, token_dim, embedding_dim=embedding_dim)
        #self.position_embedding = SinusoidalPositionEmbedding(Config.num_token, token_dim, embedding_dim)
        self.totalembedding=TotalEmbedding(token_dim, embedding_dim, channels, channels, Config.num_token, Config.dropout)
        self.transformer = TransformerEncoder(embedding_dim=embedding_dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        # 进行 Token Embedding, Convolutional Embedding 和 Position Embedding
        RF=self.totalembedding(x)
        
        # Transformer Encoder
        output = self.transformer(RF)
        
        return output

class MLP(nn.Module):
    def __init__(self, input_dim=Config.EMBEDDING_dim, hidden_dim=Config.h_dim_MLP, output_dim=Config.EMBEDDING_dim):
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

        # 调整维度以适配全连接层
        x = x.view(batch_size * channels * num_tokens, embedding_dim)  # 展平成 (batch_size * channels * num_tokens, embedding_dim)
        x = self.fc1(x)  # 第一层全连接
        x = self.relu(x)  # 激活函数
        x = self.fc2(x)  # 第二层全连接，输出维度为 64

        # 恢复形状为 (batch_size, channels, num_tokens, output_dim)
        x = x.view(batch_size, channels, num_tokens, -1)
        return x
class InverseTokenEmbedding(nn.Module):
    def __init__(self, token_embedding_layer):
        super(InverseTokenEmbedding, self).__init__()
        # 获取 TokenEmbedding 的权重并转置
        weight = token_embedding_layer.linear.weight
        self.inverse_linear = nn.Linear(weight.size(1), weight.size(0), bias=False)  # 定义逆映射层
        self.inverse_linear.weight = nn.Parameter(weight.T)  # 使用转置后的权重

    def forward(self, x):
        return self.inverse_linear(x)
def combine_segments(segments, segment_length=Config.segment_length, signal_length=Config.signal_length, overlap=Config.overlap):
    """
    将分段的信号重新组合为原始信号，并处理重叠部分。
    
    参数:
    - segments: 输入形状为 (batch_size, channels, num_segments, segment_length) 的张量
    - segment_length: 每个分段的长度，默认为 64
    - signal_length: 重组后的信号总长度，默认为 1056
    - overlap: 重叠率，默认为 50%

    返回:
    - 重组后的信号，形状为 (batch_size, channels, signal_length)
    """
    batch_size, channels, num_segments, _ = segments.shape
    step_size = int(segment_length * (1 - overlap))  # 步长，重叠 50% 的话，步长是 segment_length 的一半

    # 初始化输出张量，用于存储重新拼接后的信号
    output = torch.zeros((batch_size, channels, signal_length), dtype=segments.dtype)

    # 初始化一个计数器张量，用于记录每个位置被覆盖的次数
    counter = torch.zeros((batch_size, channels, signal_length), dtype=segments.dtype)

    # 将每个 segment 拼接到输出张量中
    for i in range(num_segments):
        start = i * step_size  # 计算每个分段的起始位置
        end = start + segment_length
        output=output.to(device)
        segments=segments.to(device)
        counter=counter.to(device)
        # 将当前分段添加到输出张量
        output[:, :, start:end] += segments[:, :, i, :]
        
        # 计数器记录每个位置被覆盖的次数
        counter[:, :, start:end] += 1

    # 处理重叠区域：将那些被多次覆盖的部分除以覆盖次数（即重叠部分除以 2）
    output = output / counter

    return output
class ChannelMerger(nn.Module):
    def __init__(self, input_channels=Config.channels, output_channels=Config.n_modes, signal_length=Config.signal_length):
        super(ChannelMerger, self).__init__()
        # 定义 Conv1d 卷积层，将输入 8 个通道的信号变换为 1 个输出信号
        # kernel_size=3, padding=1 用于保持输入输出信号长度不变
        self.conv1d = nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # 输入的 x 形状为 (batch_size, input_channels=8, signal_length=1056)
        # 通过 1D 卷积，将每个通道卷积成长度相同的信号
        output = self.conv1d(x)
        # 卷积的输出形状为 (batch_size, output_channels=Config.n_modes(=4), signal_length=1056)
        return output 

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

        # Step 4: 将 8 个通道的信号重建为 n_modes(=4) 个通道
        x = self.CM(x)  # 小波逆变换后形状为 (16, Config.n_modes, 1056)

        return x


class AutoEncoderModeDecomposer(nn.Module):
    def __init__(self, channels=Config.channels, token_dim=Config.segment_length, embedding_dim=Config.EMBEDDING_dim, num_heads=Config.num_heads, num_layers=Config.num_layers_T, dropout=Config.dropout):
        super(AutoEncoderModeDecomposer, self).__init__()
        self.encoder = Encoder(channels, token_dim, embedding_dim, num_heads, num_layers, dropout)
        token_embedding_instance = self.encoder.totalembedding.token_embedding
        self.inverse_token_embedding = InverseTokenEmbedding(token_embedding_instance)

        self.decoder = Decoder(self.inverse_token_embedding)
        
    
    def forward(self, x):  

        encoder_output = self.encoder(x)
        output = self.decoder(encoder_output)
        return output

























class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_E, num_layers, condition_dim, dropout=Config.dropout):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim_E = hidden_dim_E
        self.num_layers = num_layers
        self.condition_dim=condition_dim

        self.fc1 = nn.Linear(input_dim + condition_dim, hidden_dim_E[1])
        self.fc2 = nn.Linear(hidden_dim_E[1], hidden_dim_E[2])
        self.fc3 = nn.Linear(hidden_dim_E[2], hidden_dim_E[3])
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(hidden_dim_E[3], hidden_dim_E[0], num_layers, batch_first=True, dropout=dropout)
        
        self.proj_1 = nn.Linear(hidden_dim_E[1], hidden_dim_E[2])
        self.proj_2 = nn.Linear(hidden_dim_E[2], hidden_dim_E[3])

    def forward(self, x, condition,mask):
        
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [batch_size, seq_len] -> [batch_size, seq_len, 1]
        
        #添加掩码
        mask = mask.unsqueeze(-1)
        x = x * mask
        
        
        #print(f'Condition shape before unsqueeze: {condition.shape}')
        condition_repeated = condition.unsqueeze(1).repeat(1, x.size(1), 1)
        #print(f'Condition repeated shape: {condition_repeated.shape}')


        x = torch.cat([x, condition_repeated], dim=2)
        
        #print(f'Before fc1: {x.shape}')
        x = torch.relu(self.fc1(x))
        #print(f'After fc1: {x.shape}')
        
        x = self.dropout(x)

        residual = x
        #print(f'Before fc2: {x.shape}')
        x = torch.relu(self.fc2(x))
        #print(f'After fc2: {x.shape}')
        
        
        if x.size(2) != residual.size(2):
            residual = self.proj_1(residual)  # 调整 residual 的形状
            
        x = x + residual  # 第一层残差连接

        residual = x
        
        #print(f'Before fc3: {x.shape}')
        x = torch.relu(self.fc3(x))
        #print(f'After fc3: {x.shape}')

        if x.size(2) != residual.size(2):
            residual = self.proj_2(residual)  # 再次调整 residual 的形状
            

        x = x + residual  # 第二层残差连接
        
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim_E[0]).to(x.device)
        
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim_E[0]).to(x.device)
        
        out, (hn, _) = self.lstm(x, (h0, c0))
        
        
        return hn[-1]

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: [batch_size, seq_len, hidden_dim]
        # decoder_hidden: [batch_size, hidden_dim]
        
        # 计算注意力权重
        attn_weights = self.attn(encoder_outputs)  # [batch_size, seq_len, hidden_dim]

        # 使用softmax沿着seq_len维度计算注意力权重
        attn_weights = self.softmax(attn_weights.sum(dim=2))  # [batch_size, seq_len]

        # decoder_hidden 需要扩展维度以适应矩阵乘法
        attn_weights = attn_weights.unsqueeze(2)  # [batch_size, seq_len, 1]
        encoder_outputs = encoder_outputs.transpose(1, 2)  # [batch_size, hidden_dim, seq_len]

        # 使用批量矩阵乘法计算上下文向量
        context = torch.bmm(encoder_outputs, attn_weights)  # [batch_size, hidden_dim, 1]
        context = context.squeeze(2)  # [batch_size, hidden_dim]

        return context

class LSTMDecoderWithAttention(nn.Module):
    def __init__(self, hidden_dim_D, output_dim, num_layers, condition_dim, dropout=Config.dropout):
        super(LSTMDecoderWithAttention, self).__init__()
        self.hidden_dim_D = hidden_dim_D
        self.num_layers = num_layers
        
        # 注意力机制
        #self.attention = Attention(hidden_dim_D[0])

        # 线性层，用于结合注意力输出和条件输入
        self.fc1 = nn.Linear(hidden_dim_D[1] , hidden_dim_D[2])
        self.fc2 = nn.Linear(hidden_dim_D[2], hidden_dim_D[3])
        self.fc3 = nn.Linear(hidden_dim_D[3], hidden_dim_D[4])
        self.dropout = nn.Dropout(dropout)

        # LSTM层
        self.lstm = nn.LSTM(hidden_dim_D[4], self.hidden_dim_D[5], num_layers, batch_first=True, dropout=dropout)

        # 输出线性层
        self.fc4 = nn.Linear(hidden_dim_D[5], output_dim)

        # 投影层，用于匹配形状
        self.proj_1 = nn.Linear(hidden_dim_D[2], hidden_dim_D[3])
        self.proj_2 = nn.Linear(hidden_dim_D[3], hidden_dim_D[4])
    def forward(self, encoded, condition, seq_len):
        #condition_repeated = condition.unsqueeze(1).repeat(1, seq_len, 1)
        encoded = encoded.unsqueeze(1).repeat(1, seq_len, 1)
        
        

        # 使用注意力机制
        #context = self.attention(encoded, encoded)
        #print('done')
        # 拼接上下文与条件输入
        x = encoded
        
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        
        # 残差连接部分
        residual = x
        x = torch.relu(self.fc2(x))
        
      
       

        if x.size(2) != residual.size(2):
            residual = self.proj_1(residual)  # 调整 residual 的形状
           
        
        x = x + residual  # 第一层残差连接

        residual = x
       
        x = torch.relu(self.fc3(x))
        
       

        if x.size(2) != residual.size(2):
            residual = self.proj_2(residual)  # 再次调整 residual 的形状
           

        x = x + residual  # 第二层残差连接
        
        # 初始化 LSTM 的初始状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim_D[5]).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim_D[5]).to(x.device)
        
        # 通过 LSTM 层
        out, _ = self.lstm(x, (h0, c0))
        
        # 通过输出线性层
        out = self.fc4(out)
        
        return out

class MaskedConditionalAutoencoder(nn.Module):
    def __init__(self, input_dim, output_dim,hidden_dim_E,hidden_dim_D, num_layers, condition_dim, gap_size):
        super(MaskedConditionalAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim_E, num_layers, condition_dim)
        self.decoder = LSTMDecoderWithAttention(hidden_dim_D, output_dim, num_layers, condition_dim)
        self.gap_size = gap_size
    
    def forward(self, x, mask, condition,gap_start):  
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [batch_size, seq_len] -> [batch_size, seq_len, 1]

        encoded = self.encoder(x, condition,mask)
        #gap_start = (x.size(1) - self.gap_size) // Config.Position
        seq_len = self.gap_size
        decoded_gap = self.decoder(encoded, condition, seq_len)
        
        row_indices = torch.arange(x.size(0)).unsqueeze(1)  # [batch_size, 1]
        gap_indices = gap_start.unsqueeze(1) + torch.arange(self.gap_size).unsqueeze(0)  # [batch_size, gap_size]

        
        output = x.clone()
        output[row_indices, gap_indices] = decoded_gap

        return output