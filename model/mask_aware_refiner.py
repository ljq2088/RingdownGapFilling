"""Mask-aware residual refinement of frozen DGF14 outputs.
The adapter receives only observation, baseline prediction, gap mask and time.
Targets enter training losses/evaluation, never inference. Original weights stay frozen.
"""
import torch
from torch import nn
class MaskAwareRefiner(nn.Module):
 def __init__(self,width=24):
  super().__init__();self.entry=nn.Conv1d(4,width,7,padding=3)
  self.blocks=nn.ModuleList([nn.Sequential(nn.Conv1d(width,width,7,padding=3*d,dilation=d),nn.GroupNorm(4,width),nn.GELU()) for d in [1,2,4,8,16,32]])
  self.head=nn.Conv1d(width,1,7,padding=3);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
 def forward(self,observation,baseline,mask):
  # Numerical normalization operates on strains scaled by 1e22.
  scale=torch.sqrt(observation.square().mean(1,keepdim=True)).clamp_min(1e-4)
  time=torch.linspace(-1,1,observation.shape[1],device=observation.device,dtype=observation.dtype).expand_as(observation)
  features=torch.stack([observation/scale,baseline/scale,mask,time],dim=1)
  y=self.entry(features)
  for block in self.blocks:y=y+.2*block(y)
  return baseline+.2*scale*self.head(y).squeeze(1)

class RefinedDGF(nn.Module):
 """Drop-in inference wrapper. mask=1 inside gap; x is physical strain."""
 def __init__(self,baseline,refiner):
  super().__init__();self.baseline=baseline;self.refiner=refiner
 def forward(self,x,qt,mask):
  if mask.shape!=x.shape:raise ValueError('mask must have the same [batch,time] shape as observation')
  base=self.baseline(x,qt)
  return self.refiner(x*1e22,base*1e22,mask.to(dtype=x.dtype))/1e22
 @classmethod
 def from_checkpoints(cls,baseline_path,refiner_path,device='cpu'):
  from model.mymodel_5 import DenoiseMaskedGapsFiller
  base=DenoiseMaskedGapsFiller()
  # The legacy constructor uses nn.Parameter(encoder_weight.T), sharing CPU
  # storage. Training moved to CUDA before loading, so checkpoint encoder and
  # inverse weights are independent. Untie before CPU load to preserve both.
  layer=base.inverse_token_embedding.inverse_linear
  layer.weight=nn.Parameter(layer.weight.detach().clone())
  base.load_state_dict(torch.load(baseline_path,map_location='cpu',weights_only=True))
  state=torch.load(refiner_path,map_location='cpu',weights_only=True);refiner=MaskAwareRefiner(state['width']);refiner.load_state_dict(state['state_dict'])
  return cls(base,refiner).to(device).eval()
