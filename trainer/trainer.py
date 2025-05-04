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

def train_the_model(model, train_loader, val_loader, num_epochs, learning_rate, save_path, device, save_freq=10, early_stopping_patience=10,resume_checkpoint=None):
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)
    print(num_epochs)
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for targets,_,inputs, conditions,qt in train_loader:
            targets, inputs,conditions,qt = targets.to(device), inputs.to(device),conditions.to(device),qt.to(device)
            # masks,gap_start=generate_continuous_mask(targets.size(0), targets.size(1), model.gap_size)
            # masks=masks.to(device)
   
            optimizer.zero_grad()
            
            outputs = model(inputs,qt)
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
            loss=criterion(outputs, targets.squeeze(1))
            loss=loss.mean()
            if torch.isinf(loss).any():
                print(f"Inf loss detected at epoch {epoch+1}, batch {targets.shape}")
                return  # 退出训练
            #loss=loss.mean()#+smooth_loss(outputs,  gap_start, gap_end, Config.Smoothloss_coeff)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for targets,_,inputs, conditions,qt in val_loader:
                targets, inputs,conditions,qt =  targets.to(device), inputs.to(device),conditions.to(device),qt.to(device)
                
                # masks,gap_start=generate_continuous_mask(targets.size(0), targets.size(1), model.gap_size)
                # masks=masks.to(device)
                outputs = model(inputs,qt)
                #outputs_re=wavelet_reconstruct_from_channels(outputs)
                outputs=outputs.to(device).squeeze(1)
                #gap_start = (inputs.size(1) - model.gap_size) // Config.Position
                
                #gap_end = gap_start + model.gap_size
                loss =criterion(outputs, targets.squeeze(1))
                #loss=loss.mean()#+smooth_loss(outputs,  gap_start, gap_end, Config.Smoothloss_coeff)
                loss=loss.mean()
                val_loss += loss.item()# * inputs.size(0)

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
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}.pth")

def train_the_model_with_noise(model, train_loader, val_loader, num_epochs, learning_rate, save_path_with_noise, device, save_freq=10, early_stopping_patience=10):
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)
    print(num_epochs)
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    os.makedirs(os.path.dirname(save_path_with_noise), exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for targets,_,_,inputs, conditions in train_loader:
            targets, inputs,conditions = targets.to(device), inputs.to(device),conditions.to(device)
            # masks,gap_start=generate_continuous_mask(targets.size(0), targets.size(1), model.gap_size)
            # masks=masks.to(device)
   
            optimizer.zero_grad()
            
            outputs = model(inputs)
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
            loss=criterion(outputs, targets.squeeze(1))
            loss=loss.mean()
            
            #loss=loss.mean()#+smooth_loss(outputs,  gap_start, gap_end, Config.Smoothloss_coeff)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for targets,_,_,inputs, conditions in val_loader:
                targets, inputs,conditions =  targets.to(device), inputs.to(device),conditions.to(device)
                
                # masks,gap_start=generate_continuous_mask(targets.size(0), targets.size(1), model.gap_size)
                # masks=masks.to(device)
                outputs = model(inputs)
                #outputs_re=wavelet_reconstruct_from_channels(outputs)
                outputs=outputs.to(device).squeeze(1)
                #gap_start = (inputs.size(1) - model.gap_size) // Config.Position
                
                #gap_end = gap_start + model.gap_size
                loss =criterion(outputs, targets.squeeze(1))
                #loss=loss.mean()#+smooth_loss(outputs,  gap_start, gap_end, Config.Smoothloss_coeff)
                loss=loss.mean()
                val_loss += loss.item()# * inputs.size(0)

        # 计算平均验证损失
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        # 打印训练和验证损失
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss*1000:.4f}, Validation Loss: {val_loss*1000:.4f}')
        if (epoch + 1) % save_freq == 0:
            torch.save(model.state_dict(), f'{save_path_with_noise}_epoch_{epoch+1}.pth')
            print(f"Model saved to {save_path_with_noise}_epoch_{epoch+1}.pth")

        # Early stopping 检查
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
    torch.save(model.state_dict(), save_path_with_noise)
    print(f"Model saved to {save_path_with_noise}.pth")



def train_the_model_decomposition(model, train_loader, val_loader, num_epochs, learning_rate, save_path_decomposition, device, save_freq=10, early_stopping_patience=10):
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)
    print(num_epochs)
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    os.makedirs(os.path.dirname(save_path_decomposition), exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for signal,inputs,targets22,targets21,targets33,targets44, conditions in train_loader:
            signal,inputs,targets22,targets21,targets33,targets44, conditions =signal.to(device), inputs.to(device),targets22.to(device),targets21.to(device),targets33.to(device),targets44.to(device),conditions.to(device)
            # masks,gap_start=generate_continuous_mask(targets.size(0), targets.size(1), model.gap_size)
            # masks=masks.to(device)
   
            optimizer.zero_grad()
            
            outputs = model(inputs)
            outputs_split = torch.split(outputs, 1, dim=1)
            modes = [mode.to(device).squeeze(1) for mode in outputs_split] 
            # outputs_re=wavelet_reconstruct_from_channels(outputs)
           

            # 确保目标张量的形状与输出匹配
            # if targets.dim() == 2:
            #     targets = targets.unsqueeze(-1)  # [batch_size, seq_len] -> [batch_size, seq_len, 1]
            
            
            # 掩码计算，设置gap_start和gap_end  
            #gap_start = (inputs.size(1) - model.gap_size) // Config.Position
            
            #gap_end = gap_start + model.gap_size
           
             
           
        

            # 只计算掩码部分的损失
            #计划添加物理的损失如振幅衰减损失，微分方程损失
            #loss = combined_loss(outputs, targets,  gap_start, model.gap_size, Config.Smooth_coeff, Config.Loss_coeff)
            loss22=criterion(modes[0], targets22.squeeze(1))
            loss21=criterion(modes[1], targets21.squeeze(1))
            loss33=criterion(modes[2], targets33.squeeze(1))
            loss44=criterion(modes[3], targets44.squeeze(1))
            loss_reconstruction=criterion(modes[0]+modes[1]+modes[2]+modes[3], targets21+targets22+targets33+targets44)
            loss=loss22.mean()+loss21.mean()+loss33.mean()+loss44.mean()+Config.Coeff_reconstruction*loss_reconstruction.mean()
            loss=loss.mean()
            
            #loss=loss.mean()#+smooth_loss(outputs,  gap_start, gap_end, Config.Smoothloss_coeff)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for signal,inputs,targets22,targets21,targets33,targets44, conditions in val_loader:
                signal,inputs,targets22,targets21,targets33,targets44, conditions = signal.to(device),inputs.to(device),targets22.to(device),targets21.to(device),targets33.to(device),targets44.to(device),conditions.to(device)
                
                
                # masks,gap_start=generate_continuous_mask(targets.size(0), targets.size(1), model.gap_size)
                # masks=masks.to(device)
                outputs = model(inputs)
                outputs_split = torch.split(outputs, 1, dim=1)
                modes = [mode.to(device).squeeze(1) for mode in outputs_split]

                #outputs_re=wavelet_reconstruct_from_channels(outputs)
                
                #gap_start = (inputs.size(1) - model.gap_size) // Config.Position
                
                #gap_end = gap_start + model.gap_size
                loss22=criterion(modes[0], targets22.squeeze(1))
                loss21=criterion(modes[1], targets21.squeeze(1))
                loss33=criterion(modes[2], targets33.squeeze(1))
                loss44=criterion(modes[3], targets44.squeeze(1))
                loss_reconstruction=criterion(modes[0]+modes[1]+modes[2]+modes[3], signal.squeeze(1))
                loss=loss22.mean()+loss21.mean()+loss33.mean()+loss44.mean()+Config.Coeff_reconstruction*loss_reconstruction.mean()
                #loss=loss.mean()#+smooth_loss(outputs,  gap_start, gap_end, Config.Smoothloss_coeff)
                loss=loss.mean()
                val_loss += loss.item()# * inputs.size(0)

        # 计算平均验证损失
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        # 打印训练和验证损失
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss*1000:.4f}, Validation Loss: {val_loss*1000:.4f}')
        if (epoch + 1) % save_freq == 0:
            torch.save(model.state_dict(), f'{save_path_decomposition}_epoch_{epoch+1}.pth')
            print(f"Model saved to {save_path_decomposition}_epoch_{epoch+1}")

        # Early stopping 检查
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
    torch.save(model.state_dict(), save_path_decomposition)
    print(f"Model saved to {save_path_decomposition}.pth")
    # def train_the_model(model, train_loader, val_loader, num_epochs, learning_rate, save_path, device, save_freq=10, early_stopping_patience=10):
#     criterion = nn.MSELoss(reduction='none')
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)
#     print(num_epochs)
#     early_stopping = EarlyStopping(patience=early_stopping_patience)
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)

#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0
#         for targets,_,inputs, conditions in train_loader:
#             targets, inputs,conditions = targets.to(device), inputs.to(device),conditions.to(device)
#             # masks,gap_start=generate_continuous_mask(targets.size(0), targets.size(1), model.gap_size)
#             # masks=masks.to(device)
   
#             optimizer.zero_grad()
            
#             outputs = model(inputs)
#             # outputs_re=wavelet_reconstruct_from_channels(outputs)
           

#             # 确保目标张量的形状与输出匹配
#             # if targets.dim() == 2:
#             #     targets = targets.unsqueeze(-1)  # [batch_size, seq_len] -> [batch_size, seq_len, 1]
            
            
#             # 掩码计算，设置gap_start和gap_end  
#             #gap_start = (inputs.size(1) - model.gap_size) // Config.Position
            
#             #gap_end = gap_start + model.gap_size
           
      
#             outputs=outputs.to(device).squeeze(1)
        

#             # 只计算掩码部分的损失
#             #计划添加物理的损失如振幅衰减损失，微分方程损失
#             #loss = combined_loss(outputs, targets,  gap_start, model.gap_size, Config.Smooth_coeff, Config.Loss_coeff)
#             loss=criterion(outputs, targets.squeeze(1))
#             loss=loss.mean()
#             if torch.isinf(loss).any():
#                 print(f"Inf loss detected at epoch {epoch+1}, batch {targets.shape}")
#                 return  # 退出训练
#             #loss=loss.mean()#+smooth_loss(outputs,  gap_start, gap_end, Config.Smoothloss_coeff)
            
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

#             optimizer.step()
#             train_loss += loss.item()
#         train_loss /= len(train_loader.dataset)
        
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for targets,_,inputs, conditions in val_loader:
#                 targets, inputs,conditions =  targets.to(device), inputs.to(device),conditions.to(device)
                
#                 # masks,gap_start=generate_continuous_mask(targets.size(0), targets.size(1), model.gap_size)
#                 # masks=masks.to(device)
#                 outputs = model(inputs)
#                 #outputs_re=wavelet_reconstruct_from_channels(outputs)
#                 outputs=outputs.to(device).squeeze(1)
#                 #gap_start = (inputs.size(1) - model.gap_size) // Config.Position
                
#                 #gap_end = gap_start + model.gap_size
#                 loss =criterion(outputs, targets.squeeze(1))
#                 #loss=loss.mean()#+smooth_loss(outputs,  gap_start, gap_end, Config.Smoothloss_coeff)
#                 loss=loss.mean()
#                 val_loss += loss.item()# * inputs.size(0)

#         # 计算平均验证损失
#         val_loss /= len(val_loader.dataset)
#         scheduler.step(val_loss)
#         # 打印训练和验证损失
#         print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss*1000:.4f}, Validation Loss: {val_loss*1000:.4f}')
#         if (epoch + 1) % save_freq == 0:
#             torch.save(model.state_dict(), f'{save_path}_epoch_{epoch+1}.pth')
#             print(f"Model saved to {save_path}_epoch_{epoch+1}")

#         # Early stopping 检查
#         early_stopping(val_loss)
#         if early_stopping.early_stop:
#             print(f"Early stopping at epoch {epoch+1}")
#             break
#     torch.save(model.state_dict(), save_path)
#     print(f"Model saved to {save_path}.pth")