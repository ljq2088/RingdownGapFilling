import torch
import torch.nn.functional as F
import torchaudio.transforms as transforms
import torch
import torch.nn as nn
# 定义 STFT 转换
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stft_transform = transforms.Spectrogram(n_fft=512, win_length=400, hop_length=160).to(device)

# 计算频域损失的函数
def stft_loss(predicted, target):
    
    
    # 执行 STFT 转换
    pred_stft = stft_transform(predicted.squeeze(-1))
    target_stft = stft_transform(target.squeeze(-1))
    
    
    
    
    # 确保 STFT 结果不包含 NaN
    if torch.isnan(pred_stft).any() or torch.isnan(target_stft).any():
        print("STFT contains NaN values.")
    
    # 返回损失
    return F.mse_loss(pred_stft, target_stft)  # 可以选择 MSE 或 L1 损失


def smooth_loss(outputs,  gap_start, gap_end, coeff):
    

    # 平滑性损失：计算掩码两端的信号差异
    smoothness_loss = 0

    # 平滑性损失的计算
    if gap_start > 0:
        left_edge_difference = (outputs[:, gap_start, :] - outputs[:, gap_start - 1, :]).pow(2).mean()
        smoothness_loss += coeff[0]*left_edge_difference
        left_derivative_loss=(outputs[:, gap_start, :]*2 - outputs[:, gap_start - 1, :]-outputs[:,gap_start+1,:]).pow(2).mean()
        smoothness_loss+= coeff[1]*left_derivative_loss   
    
    if gap_end < outputs.size(1)-1:
        right_edge_difference = (outputs[:, gap_end - 1, :] - outputs[:, gap_end, :]).pow(2).mean()
        smoothness_loss += coeff[0]*right_edge_difference
        right_derivative_loss=(outputs[:, gap_end, :]*2 - outputs[:, gap_end - 1, :]-outputs[:,gap_end+1,:]).pow(2).mean()
        smoothness_loss+= coeff[1]*right_derivative_loss

    smooth_loss =   smoothness_loss

    return smooth_loss

# def combined_loss(outputs, targets,  gap_start, gap_end, coeff, Loss_coeff):
#     # 时域损失
#     mse_loss = nn.MSELoss(reduction='none')
    
#     # 频域损失
#     #stft_loss_value = stft_loss(outputs, targets)
    
#     # 平滑损失（如果需要）
#     #smooth_loss_value = smooth_loss(outputs,  gap_start, gap_end, coeff)
#     mask = torch.zeros_like(targets)
#     mask[:, gap_start:gap_end, :] = 1
    
#     # 复合损失
#     total_loss = (mask*mse_loss(outputs, targets)).mean() #+ Loss_coeff[0] * (mask*stft_loss_value).mean() + Loss_coeff[1] * smooth_loss_value
#     return total_loss
def combined_loss(outputs, targets, gap_start, gap_size, coeff, Loss_coeff):
    # 时域损失
    mse_loss = nn.MSELoss(reduction='none')
    
    try:
        mse_value = mse_loss(outputs, targets)
        if torch.isnan(mse_value).any():
            print("MSE Loss contai ns NaN")
        
        #频域损失
        stft_loss_value = stft_loss(outputs, targets)
        if torch.isnan(stft_loss_value).any():
            print("STFT Loss contains NaN")
        
        # 平滑损失
        # smooth_loss_value = smooth_loss(outputs, gap_start, gap_end, coeff)
        # if torch.isnan(smooth_loss_value).any():
        #     print("Smooth Loss contains NaN")

        # 损失加权
        row_indices = torch.arange(targets.size(0)).unsqueeze(1)  # [batch_size, 1]
        gap_indices = gap_start.unsqueeze(1) + torch.arange(gap_size).unsqueeze(0)  # [batch_size, gap_size]

        mask = torch.zeros_like(targets)
        mask[row_indices, gap_indices] = 1
        
        total_loss = (mask * mse_value).mean()# + Loss_coeff[0] * stft_loss_value.mean() #+ Loss_coeff[1] * smooth_loss_value
        return total_loss
    except Exception as e:
        print(f"Error in combined_loss: {e}")
        raise e
