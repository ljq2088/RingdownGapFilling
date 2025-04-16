from config.config import Config
import numpy as np
from data.data_generation import generate_data
from data.data_generation import generate_test_data
from data.data_generation import generate_data_with_noise
from data.data_generation import generate_data_for_mode_decomposition
from data.data_generation import combine_data
from data.data_generation import *
import os
# signals, conditions = generate_data(
#         Config.num_samples
#     )
# 定义保存路径和临时文件路径
SAVE_PATH_1= 'data/signal_data.npz'
SAVE_PATH_2 = 'data/signal_test_data.npz'
SAVE_PATH_noise= 'data/signal_data_with_noise.npz'
SAVE_PATH_noise_SNR5= 'data/signal_data_with_noise_SNR5.npz'
SAVE_PATH_noise_SNR5_signals= 'data/signal_data_with_noise_SNR5_signals.npz'
TEMP_DIR_1 = 'data/temp_files'
TEMP_DIR_2 = 'data/temp_files2'
TEMP_DIR_3 = 'data/temp_files3'
TEMP_DIR_4 = 'data/temp_files4'
#检查文件夹是否存在，不存在则创建，存在则清空并重新创建
if not os.path.exists(TEMP_DIR_4):
    os.makedirs(TEMP_DIR_4)
else:
    for file in os.listdir(TEMP_DIR_4):
        file_path = os.path.join(TEMP_DIR_4, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

signals, noises,datas,conditions = generate_data(Config.num_samples,TEMP_DIR=TEMP_DIR_4,SAVE_PATH=SAVE_PATH_noise_SNR5_signals)
#combine_data()
# signals,noises,datas, conditions = generate_data_with_noise(
#         Config.num_samples
#     )
# np.savez('data/signal_data_with_noise.npz', signals=signals, noises=noises,datas=datas, conditions=conditions)
# signals,signals_22,signals_21,signals_33,signals_44, conditions = generate_data_for_mode_decomposition(
#         Config.num_samples
#     )
# np.savez('data/signal_data_for_mode_decomposition.npz', signals=signals, signals_22=signals_22,signals_21=signals_21,signals_33=signals_33,signals_44=signals_44, conditions=conditions)