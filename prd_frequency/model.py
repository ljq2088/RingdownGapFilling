"""Original DGF with a trainable physical-band feature path past saturated ReLUs."""
import torch
from torch import nn
from torch.nn import functional as F
from config.config import Config
from prd_optimize.model import OptimizedDGF
from prd_physical.common import sha
from .settings import START_CHECKPOINT,START_SHA
class FrequencyAdapter(nn.Module):
 def __init__(self):
  super().__init__()
  self.encoder=nn.Sequential(nn.Conv2d(2,8,3,stride=2,padding=1),nn.GroupNorm(2,8),nn.GELU(),nn.Conv2d(8,16,3,stride=2,padding=1),nn.GroupNorm(4,16),nn.GELU(),nn.Conv2d(16,16,3,stride=2,padding=1),nn.GroupNorm(4,16),nn.GELU())
  self.head=nn.Conv2d(16,Config.channels,1);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
 def forward(self,low):
  low=low.reshape(-1,2,16,264)
  features=torch.stack([torch.log1p(low[:,0]),low[:,1]/torch.pi],dim=1)
  return .2*self.head(self.encoder(features))
class FrequencyDGF(nn.Module):
 def __init__(self,variant):
  super().__init__();self.variant=variant
  assert sha(START_CHECKPOINT)==START_SHA
  self.backbone=OptimizedDGF('joint_hann')
  self.backbone.load_state_dict(torch.load(START_CHECKPOINT,map_location='cpu',weights_only=True)['state_dict'])
  self.frequency=FrequencyAdapter() if variant=='frequency_adapter' else None
 def input_bands(self,u):return self.backbone.input_bands(u)
 def time_frequency_features(self,q,low):
  m=self.backbone.dgf.base.encoder.QtranTimeMixer
  x=m.mlp_time(q);x=x.permute(0,2,3,1);x=m.mlp_channel(x);x=x.permute(0,3,2,1)
  x=m.mlp_feature(x);x=x.permute(0,1,3,2)
  e=m.embedding
  x=F.relu(e.conv1(x));x=F.relu(e.conv2(x));x=F.relu(e.conv3(x))
  residual=self.frequency(low)
  assert x.shape==residual.shape
  x=x+residual
  x=e.fc1(x.view(x.size(0),-1));x=x.view(-1,m.channels,m.num_token,m.token_dim)
  x=m.mlp_token_dim(x);x=x.permute(0,1,3,2);x=m.mlp_num_token(x);x=x.permute(0,3,2,1)
  x=m.mlp_channel_2(x);return x.permute(0,3,1,2)
 def forward(self,u,q,mask,bands=None,q_low=None):
  if self.frequency is None:return self.backbone(u,q,mask,bands)
  if q_low is None:raise ValueError('frequency variant requires observed-input physical-band CWT features')
  b,t=u.shape[0],u.shape[-1];x=u.reshape(b*2,t)
  if bands is None:bands=self.input_bands(u)
  base=self.backbone.dgf.base
  _,mean,std=base.nor((x/1e22)*Config.Amp_factor**2)
  features=base.seg(bands.reshape(b*2,8,t))
  qt=self.time_frequency_features(q.reshape(b*2,2,16,264),q_low)
  encoded=base.encoder.transformer(base.encoder.totalembedding(features,qt))
  out=base.decoder(encoded).squeeze(1)
  p=(out*std.unsqueeze(1)+mean.unsqueeze(1))/Config.Amp_factor**2*1e22
  m=mask[:,None,:].expand(-1,2,-1).reshape(b*2,t)
  p=self.backbone.dgf.refiner(x,p,m).reshape(b,2,t)
  return self.backbone.joint(u,p,mask)
