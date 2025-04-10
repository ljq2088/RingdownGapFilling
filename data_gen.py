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
TEMP_DIR_1 = 'data/temp_files'
TEMP_DIR_2 = 'data/temp_files2'
TEMP_DIR_3 = 'data/temp_files3'
signals, noises,datas,conditions = generate_data(Config.num_samples,TEMP_DIR=TEMP_DIR_3,SAVE_PATH=SAVE_PATH_noise)
#combine_data()
# signals,noises,datas, conditions = generate_data_with_noise(
#         Config.num_samples
#     )
# np.savez('data/signal_data_with_noise.npz', signals=signals, noises=noises,datas=datas, conditions=conditions)
# signals,signals_22,signals_21,signals_33,signals_44, conditions = generate_data_for_mode_decomposition(
#         Config.num_samples
#     )
# np.savez('data/signal_data_for_mode_decomposition.npz', signals=signals, signals_22=signals_22,signals_21=signals_21,signals_33=signals_33,signals_44=signals_44, conditions=conditions)