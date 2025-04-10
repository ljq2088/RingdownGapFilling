import torch
import torch.nn as nn
from config.config import Config

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, condition_dim, dropout=Config.dropout):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fc1 = nn.Linear(input_dim + condition_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, condition,mask):
        
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [batch_size, seq_len] -> [batch_size, seq_len, 1]
        
        #添加掩码
        mask = mask.unsqueeze(-1)
        x = x * mask
        
        
        condition_repeated = condition.unsqueeze(1).repeat(1, x.size(1), 1)
        x = torch.cat([x, condition_repeated], dim=2)
       

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)

        residual = x
        x = torch.relu(self.fc2(x))
       

        if x.size(2) != residual.size(2):
            residual = self.proj(residual)  # 调整 residual 的形状
            
        x = x + residual  # 第一层残差连接

        residual = x
        x = torch.relu(self.fc3(x))
        

        if x.size(2) != residual.size(2):
            residual = self.proj(residual)  # 再次调整 residual 的形状
            

        x = x + residual  # 第二层残差连接

        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

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
    def __init__(self, hidden_dim, output_dim, num_layers, condition_dim, dropout=Config.dropout):
        super(LSTMDecoderWithAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 注意力机制
        self.attention = Attention(hidden_dim)

        # 线性层，用于结合注意力输出和条件输入
        self.fc1 = nn.Linear(hidden_dim + condition_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # LSTM层
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # 输出线性层
        self.fc4 = nn.Linear(hidden_dim, output_dim)

        # 投影层，用于匹配形状
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, encoded, condition, seq_len):
        condition_repeated = condition.unsqueeze(1).repeat(1, seq_len, 1)
        encoded = encoded.unsqueeze(1).repeat(1, seq_len, 1)

        # 使用注意力机制
        context = self.attention(encoded, encoded)
        
        # 拼接上下文与条件输入
        x = torch.cat([context.unsqueeze(1).repeat(1, seq_len, 1), condition_repeated], dim=2)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)

        # 残差连接部分
        residual = x
        x = torch.relu(self.fc2(x))
        
      
       

        if x.size(2) != residual.size(2):
            residual = self.proj(residual)  # 调整 residual 的形状
           
        
        x = x + residual  # 第一层残差连接

        residual = x
        x = torch.relu(self.fc3(x))

       

        if x.size(2) != residual.size(2):
            residual = self.proj(residual)  # 再次调整 residual 的形状
           

        x = x + residual  # 第二层残差连接

        # 初始化 LSTM 的初始状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # 通过 LSTM 层
        out, _ = self.lstm(x, (h0, c0))

        # 通过输出线性层
        out = self.fc4(out)
        
        return out


class MaskedConditionalAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, condition_dim, gap_size):
        super(MaskedConditionalAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, num_layers, condition_dim)
        self.decoder = LSTMDecoderWithAttention(hidden_dim, input_dim, num_layers, condition_dim)
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