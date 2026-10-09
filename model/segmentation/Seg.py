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

from utils.segment import *


class Segmentation(nn.Module):
    def __init__(self):
        super(Segmentation, self).__init__()
    def seg(self, x):
        return segment_signal_gpu(x)

    def forward(self, x):
        x = self.seg(x)
        return x







