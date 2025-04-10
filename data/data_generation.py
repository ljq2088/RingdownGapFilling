import numpy as np
from config.config import Config
from .waveform import *
from .ringdown_waveform import Gap_dir as Ga
import torch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from utils.psd import PSD_Lisa_no_Response
from utils.noise import fftfilt, stat_gauss_noise,generate_noise_from_psd
from scipy.fftpack import fft, ifft
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import firwin2,welch
import os
# def rotate_list(lst, position=2):
#     index = len(lst) // position
#     return lst[index:] + lst[:index]
SAVE_PATH_1= 'data/signal_data.npz'
SAVE_PATH_2 = 'data/signal_test_data.npz'
SAVE_PATH_noise= 'data/signal_data_with_noise.npz'
TEMP_DIR_1 = 'data/temp_files'
TEMP_DIR_2 = 'data/temp_files2'
os.makedirs(TEMP_DIR_1, exist_ok=True)  # 创建临时文件目录
scale=Config.scale
samp_freq=Config.samp_freq
N=round(samp_freq/Config.f_step)
time_vec=1/samp_freq*np.arange(0,N,1)
Noises=True
def generate_single_data(i):
    # 生成单个数据的代码
    Mtot = np.random.uniform(Config.parameters[0], Config.parameters[1])
    M_ratio = np.random.uniform(Config.parameters[2], Config.parameters[3])
    R_shift = np.random.uniform(Config.parameters[4], Config.parameters[5])
    signal_length = Config.signal_length

    para = [Mtot, M_ratio, R_shift]
    freq_ifft = np.arange(Config.f_in, Config.f_out, Config.f_step)
    f_sf = sf(freq_ifft, para, para_dw, para_dtau)
    st = Ga.Freq_ifft(f_sf)

    if Noises:
        
        index = int(1/2*len(st))
        st=np.concatenate((st[index:], st[:index]))
        PSD=PSD_Lisa_no_Response(freq_ifft)
        out_noise, _ = generate_noise_from_psd(N,freq_ifft,PSD, sample_rate=samp_freq)
        
        #print(len(out_noise[0]))
        start1=int(1/2*(len(out_noise[0])-signal_length))
        start2=int(1/2*(len(st)-signal_length))
        #print(len(out_noise[0][start:start+signal_length]))
        signal = st[start2:start2+signal_length]
        signal=torch.tensor(signal)
        signal=torch.real(signal)
        
        noise=out_noise[0][start1:start1+signal_length]
        noise=torch.tensor(noise)
        noise=torch.real(noise)

        data=signal+noise
        return signal,noise,data, torch.tensor([Mtot, M_ratio, R_shift], dtype=torch.float32)

    else:
        original_signal = st[:signal_length]
        original_signal = torch.tensor(original_signal, dtype=torch.float32)
        original_signal = torch.real(original_signal)
        return original_signal, torch.tensor([Mtot, M_ratio, R_shift], dtype=torch.float32)

def generate_data(num_samples,TEMP_DIR=TEMP_DIR_1, SAVE_PATH=SAVE_PATH_noise):
   
    """生成指定数量的样本数据，支持断点续传
    
    Args:
        num_samples: 需要生成的总样本数量
        
    Returns:
        Tuple: (signals, conditions) 全部样本数据
    """
    # 1. 初始化临时目录
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # 2. 准确计算已有样本数（忽略非样本文件）
    existing_samples = len([
        f for f in os.listdir(TEMP_DIR) 
        if f.startswith('sample_') and f.endswith('.npz')
    ])
    
    # 3. 计算需要生成的数量
    start_index = existing_samples
    remaining_samples = max(0, num_samples - start_index)
    
    # 4. 显示进度信息
    print(f"当前进度: {start_index}/{num_samples} | 待生成: {remaining_samples}")

    # 5. 生成缺失样本
    if remaining_samples > 0:
        try:
            with Pool(cpu_count()) as pool:
                results = tqdm(
                    pool.imap_unordered(generate_single_data, range(start_index, num_samples)),
                    #pool.imap_unordered(generate_single_data_with_noise, range(start_index, num_samples)),
                    total=remaining_samples,
                    desc="生成样本"
                )
                if Noises:
                    print('generate data with noise')
                    for i, (signal, noise, data, condition) in enumerate(results):
                        # 6. 实时保存每个样本
                        np.savez(
                            os.path.join(TEMP_DIR, f'sample_{start_index + i}.npz'),
                            signal=signal.numpy(),
                            noise=noise.numpy(),
                            data=data.numpy(),
                            condition=condition.numpy()
                        )
                else:
                    print('generate data without noise')
                    for i, (signal, condition) in enumerate(results):
                        # 6. 实时保存每个样本
                        np.savez(
                            os.path.join(TEMP_DIR, f'sample_{start_index + i}.npz'),
                            signal=signal.numpy(),
                            condition=condition.numpy()
                    )
        except Exception as e:
            print(f"生成过程中出错: {str(e)}")
            raise RuntimeError("数据生成失败，请检查参数设置")
    
    # 7. 返回当前内存中的所有数据（可选）
    # 注意：对于大数据集建议使用 combine_data() 单独处理
    return combine_data(TEMP_DIR, SAVE_PATH)  # 或者 return None 仅执行生成操作

def combine_data(TEMP_DIR=TEMP_DIR_1, SAVE_PATH=SAVE_PATH_1):
    """合并所有样本数据并返回完整数据集"""
    try:
        # 获取并按序号排序样本文件
        sample_files = sorted(
            [f for f in os.listdir(TEMP_DIR) if f.startswith('sample_') and f.endswith('.npz')],
            key=lambda x: int(x.split('_')[1].split('.')[0])
        )
        
        if not sample_files:
            raise ValueError("未找到任何样本文件")
        if Noises:
            signals, noises,datas, conditions = [], [],[], []
            for f in tqdm(sample_files, desc="加载样本"):
                data = np.load(os.path.join(TEMP_DIR, f))
                signals.append(torch.tensor(data['signal']))
                noises.append(torch.tensor(data['noise']))
                datas.append(torch.tensor(data['data']))
                conditions.append(torch.tensor(data['condition']))
            
            # 合并数据
            signals = torch.stack(signals)
            noises = torch.stack(noises)
            datas = torch.stack(datas)
            conditions = torch.stack(conditions)
            
            # 保存完整数据集
            np.savez(SAVE_PATH, signals=signals.numpy(), noises=noises.numpy(),datas=datas.numpy(), conditions=conditions.numpy())
            print(f"已合并 {len(signals)} 个样本到 {SAVE_PATH}")
            
            return signals, noises,datas, conditions
        else:
            signals, conditions = [], []
            for f in tqdm(sample_files, desc="加载样本"):
                data = np.load(os.path.join(TEMP_DIR, f))
                signals.append(torch.tensor(data['signal']))
                conditions.append(torch.tensor(data['condition']))
            
            # 合并数据
            signals = torch.stack(signals)
            conditions = torch.stack(conditions)
            
            # 保存完整数据集
            np.savez(SAVE_PATH, signals=signals.numpy(), conditions=conditions.numpy())
            print(f"已合并 {len(signals)} 个样本到 {SAVE_PATH}")
            
            return signals, conditions
        
    except Exception as e:
        print(f"合并数据时出错: {str(e)}")
        raise

def generate_test_single_data(i):
    # 生成单个数据的代码
    Mtot =  5e5
    M_ratio = 0.5
    R_shift = 1
    signal_length = Config.signal_length

    para = [Mtot, M_ratio, R_shift]
    freq_ifft = np.arange(Config.f_in, Config.f_out, Config.f_step)
    f_sf = sf(freq_ifft, para, para_dw, para_dtau)
    st = Ga.Freq_ifft(f_sf)
    original_signal = st[:signal_length]
    original_signal=torch.tensor(original_signal)
    original_signal=torch.real(original_signal)
    return torch.tensor(original_signal), torch.tensor([Mtot, M_ratio, R_shift])

def generate_test_data(num_samples):
    # 使用 Pool 进行并行计算并显示进度条
    with Pool(cpu_count()) as p:
        results = list(tqdm(p.imap(generate_test_single_data, range(num_samples)), total=num_samples))
    
    signals, conditions = zip(*results)
    signals = torch.stack(signals)
    conditions = torch.stack(conditions)
    
    return signals, conditions

def generate_single_data_with_noise(i):
    # 生成单个数据的代码
    Mtot = np.random.uniform(Config.parameters[0], Config.parameters[1])
    M_ratio = np.random.uniform(Config.parameters[2], Config.parameters[3])
    R_shift = np.random.uniform(Config.parameters[4], Config.parameters[5])
    signal_length = Config.signal_length_before_whitened

    
    para = [Mtot, M_ratio, R_shift]
    freq = np.arange(Config.f_in, Config.f_out+Config.f_step, Config.f_step)
    f_sf = sf(freq, para, para_dw, para_dtau)
    st = Ga.Freq_ifft(f_sf)*scale

    index = int(1/2*len(st))
    st=np.concatenate((st[index:], st[:index]))
    PSD=PSD_Lisa_no_Response(freq)
    out_noise, _ = generate_noise_from_psd(N,freq,PSD, sample_rate=samp_freq)
    
    #print(len(out_noise[0]))
    start1=int(1/2*(len(out_noise[0])-signal_length))
    start2=int(1/2*(len(st)-signal_length))
    #print(len(out_noise[0][start:start+signal_length]))
    signal = st[start2:start2+signal_length]
    signal=torch.tensor(signal)
    signal=torch.real(signal)
    
    noise=out_noise[0][start1:start1+signal_length]
    noise=torch.tensor(noise)
    noise=torch.real(noise)

    data=signal+noise
    return signal.clone().detach(),noise.clone().detach(),data.clone().detach(), torch.tensor([Mtot, M_ratio, R_shift])

def generate_data_with_noise(num_samples):
    # 使用 Pool 进行并行计算并显示进度条
    with Pool(cpu_count()) as p:
        results = list(tqdm(p.imap(generate_single_data_with_noise, range(num_samples)), total=num_samples))
    
    signals,noises,datas, conditions = zip(*results)
    signals = torch.stack(signals)
    noises = torch.stack(noises)
    datas = torch.stack(datas)
    conditions = torch.stack(conditions)
    
    return signals,noises,datas, conditions

def generate_single_data_for_mode_decomposition(i):
    # 生成单个数据的代码
    Mtot = np.random.uniform(Config.parameters[0], Config.parameters[1])
    M_ratio = np.random.uniform(Config.parameters[2], Config.parameters[3])
    R_shift = np.random.uniform(Config.parameters[4], Config.parameters[5])
    signal_length = Config.signal_length

    para = [Mtot, M_ratio, R_shift]
    freq_ifft = np.arange(Config.f_in, Config.f_out, Config.f_step)
    f_sf22, f_sf21,f_sf33, f_sf44 = sf_decomposition(freq_ifft, para, para_dw, para_dtau)
    f_sf = f_sf22 + f_sf21 + f_sf33 + f_sf44
    st = Ga.Freq_ifft(f_sf)
    st22=Ga.Freq_ifft(f_sf22)
    st21=Ga.Freq_ifft(f_sf21)
    st33=Ga.Freq_ifft(f_sf33)  
    st44=Ga.Freq_ifft(f_sf44)

    original_signal = st[:signal_length]
    original_signal=torch.tensor(original_signal)
    original_signal=torch.real(original_signal)

    signal_22 = st22[:signal_length]
    signal_22=torch.tensor(signal_22)
    signal_22=torch.real(signal_22)

    signal_21 = st21[:signal_length]
    signal_21=torch.tensor(signal_21)
    signal_21=torch.real(signal_21)

    signal_33 = st33[:signal_length]
    signal_33=torch.tensor(signal_33)
    signal_33=torch.real(signal_33)

    signal_44 = st44[:signal_length]
    signal_44=torch.tensor(signal_44)
    signal_44=torch.real(signal_44)
    return torch.tensor(original_signal),torch.tensor(signal_22),torch.tensor(signal_21),torch.tensor(signal_33),torch.tensor(signal_44),torch.tensor([Mtot, M_ratio, R_shift])

def generate_data_for_mode_decomposition(num_samples):
    # 使用 Pool 进行并行计算并显示进度条
    with Pool(cpu_count()) as p:
        results = list(tqdm(p.imap(generate_single_data_for_mode_decomposition, range(num_samples)), total=num_samples))
    
    signals,signals_22,signals_21,signals_33,signals_44, conditions = zip(*results)
    signals = torch.stack(signals)
    signals_22 = torch.stack(signals_22)    
    signals_21 = torch.stack(signals_21)
    signals_33 = torch.stack(signals_33)
    signals_44 = torch.stack(signals_44)
    conditions = torch.stack(conditions)
    
    return signals, signals_22,signals_21,signals_33,signals_44,conditions