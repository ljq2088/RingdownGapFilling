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

from utils.wavelet import *



class MyWavelet_GPU(nn.Module):
    def __init__(self):
        super(MyWavelet_GPU, self).__init__()
        # self.wavelet = wavelet  # 小波选择
        # self.scales = np.linspace(1, scales, num=scales)  # 将scales转为一个线性数组 # 小波变换的尺度数 
        #self.wav=gpu_wavelet_bandpass
    def wav(self,x):
        return wavelet_bandpass_gpu(x)
    def rewav(self,x):
        return wavelet_reconstruct_gpu(x)
    def forward(self, x):
        #检查形状
        if x.shape != self.wav(x).shape:
            raise ValueError(f"Input shape {x.shape} does not match output shape {wavelet_bandpass_gpu(x).shape}")
        return self.wav(x)








