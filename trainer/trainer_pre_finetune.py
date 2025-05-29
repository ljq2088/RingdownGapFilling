
import torch
import torch.nn as nn
import torch.optim as optim
import os
from config.config import Config
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.loss import combined_loss
from utils.mask import generate_continuous_mask
from utils.wavelet import wavelet_reconstruct_from_channels
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def visualize_prediction(model, dataloader, device, epoch, save_dir="./viz"):
    model.eval()
    with torch.no_grad():
        inputs_batch, mask_batch, targets_batch, _, qt_batch = next(iter(dataloader))
        inputs_batch = inputs_batch.to(device)
        mask_batch = mask_batch.to(device)
        targets_batch = targets_batch.to(device)
        qt_batch = qt_batch.to(device)

        outputs = model(inputs_batch, qt_batch, mask_batch)
        outputs = outputs.squeeze(1).cpu().numpy()
        inputs = inputs_batch.squeeze(1).cpu().numpy()
        targets = targets_batch.squeeze(1).cpu().numpy()

        for i in range(3):  # 画前3个样本
            plt.figure(figsize=(10, 4))
            plt.plot(targets[i], label='Target', linewidth=1.5)
            plt.plot(inputs[i], label='Input (Masked+Noise)', alpha=0.7)
            plt.plot(outputs[i], label='Prediction', linestyle='--')
            plt.title(f"Sample {i+1} - Epoch {epoch}")
            plt.xlabel("Time step")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{save_dir}/epoch{epoch}_sample{i}.png")
            plt.close()


def save_checkpoint(path, epoch, model, optimizer, scheduler, early_stopping):
    """<<< CHECKPOINT >>>: 保存完整训练状态"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'early_stopping': {
            'best_loss': early_stopping.best_loss,
            'counter': early_stopping.counter
        }
    }, path)
    print(f"Checkpoint saved to {path}")

# Early stopping class implementation
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        """
        patience: 允许的验证损失不减少的最大 epoch 数
        min_delta: 验证损失减少的最小值，小于此值不算作改善
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
from transformers import get_linear_schedule_with_warmup
def compute_weighted_loss(pred, target, mask_1d, alpha=5.0, l1_weight=1.0):
    weight = torch.ones_like(mask_1d, dtype=torch.float32, device=pred.device)
    weight[mask_1d.bool()] = alpha
    
    mse = nn.functional.mse_loss(pred, target, reduction='none')  # [B, 1056]
    l1  = nn.functional.l1_loss(pred, target, reduction='none')   # [B, 1056]
    loss = (mse + l1_weight * l1) * weight
    return loss.mean()




def train_the_model(model, train_loader, val_loader, num_epochs, learning_rate, save_path, device, save_freq=10, early_stopping_patience=10,resume_checkpoint=None):
    
    if Config.stage == "pretrain":
        print("🔧 Entering pretraining stage...")
    elif Config.stage == "finetune":
        print("🔁 Entering finetuning stage...")
        model.transfer_pretrained_decoder()
    else:
        raise ValueError(f"Unknown stage: {Config.stage}")
    criterion = nn.MSELoss(reduction='none')
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    # 替换旧的 scheduler 初始化：
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=Config.weight_decay)
    num_training_steps = num_epochs * len(train_loader)
    warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    print(num_epochs)
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(num_epochs):
        l1_weight = Config.L1_weight * (epoch / num_epochs)  # L1 权重线性增加
        model.train()
        train_loss = 0
        for targets,mask_1d,inputs, conditions,qt in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            targets, inputs,conditions,qt = targets.to(device), inputs.to(device),conditions.to(device),qt.to(device)
            # masks,gap_start=generate_continuous_mask(targets.size(0), targets.size(1), model.gap_size)
            # masks=masks.to(device)
            mask_1d=mask_1d.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs, qt, mask_1d)   # 注意 mask_1d 作为输入
            
            # outputs_re=wavelet_reconstruct_from_channels(outputs)
           

            # 确保目标张量的形状与输出匹配
            # if targets.dim() == 2:
            #     targets = targets.unsqueeze(-1)  # [batch_size, seq_len] -> [batch_size, seq_len, 1]
            
            
            # 掩码计算，设置gap_start和gap_end  
            #gap_start = (inputs.size(1) - model.gap_size) // Config.Position
            
            #gap_end = gap_start + model.gap_size
           
      
            outputs=outputs.to(device).squeeze(1)
        

            # 只计算掩码部分的损失
            #计划添加物理的损失如振幅衰减损失，微分方程损失
            #loss = combined_loss(outputs, targets,  gap_start, model.gap_size, Config.Smooth_coeff, Config.Loss_coeff)
            
            loss = compute_weighted_loss(outputs, targets.squeeze(1), mask_1d, alpha=Config.loss_alpha)
            loss=loss.mean()
            if torch.isinf(loss).any():
                print(f"Inf loss detected at epoch {epoch+1}, batch {targets.shape}")
                return  # 退出训练
            #loss=loss.mean()#+smooth_loss(outputs,  gap_start, gap_end, Config.Smoothloss_coeff)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.detach().item()

        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for targets,mask_1d,inputs, conditions,qt in val_loader:
                targets, inputs,conditions,qt =  targets.to(device), inputs.to(device),conditions.to(device),qt.to(device)
                mask_1d=mask_1d.to(device)
                # masks,gap_start=generate_continuous_mask(targets.size(0), targets.size(1), model.gap_size)
                # masks=masks.to(device)
                outputs = model(inputs,qt,mask_1d)
                #outputs_re=wavelet_reconstruct_from_channels(outputs)
                outputs=outputs.to(device).squeeze(1)
                #gap_start = (inputs.size(1) - model.gap_size) // Config.Position
                
                #gap_end = gap_start + model.gap_size
                loss = compute_weighted_loss(outputs, targets.squeeze(1), mask_1d, alpha=Config.loss_alpha)
                #loss=loss.mean()#+smooth_loss(outputs,  gap_start, gap_end, Config.Smoothloss_coeff)
                loss=loss.mean()
                val_loss += loss.detach().item()


        # 计算平均验证损失
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        # 打印训练和验证损失
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss*1000:.4f}, Validation Loss: {val_loss*1000:.4f}')
        if (epoch + 1) % save_freq == 0:
            torch.save(model.state_dict(), f'{save_path}_epoch_{epoch+1}.pth')
            print(f"Model saved to {save_path}_epoch_{epoch+1}")

        # Early stopping 检查
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
        # if (epoch + 1) % 5 == 0:  # 每5个epoch保存一次图像
        #     visualize_prediction(model, val_loader, device, epoch)

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}.pth")

