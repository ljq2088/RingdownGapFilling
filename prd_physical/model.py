"""Original DGF plus its small mask residual adapter, in calibrated working units."""
import torch
from torch import nn
from config.config import Config
from model.mymodel_5 import DenoiseMaskedGapsFiller
from model.mask_aware_refiner import MaskAwareRefiner

class PhysicalDGF(nn.Module):
 def __init__(self,checkpoint):
  super().__init__();self.base=DenoiseMaskedGapsFiller()
  layer=self.base.inverse_token_embedding.inverse_linear
  # Preserve independently trained encoder/inverse weights on CPU too.
  layer.weight=nn.Parameter(layer.weight.detach().clone())
  self.base.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
  self.refiner=MaskAwareRefiner(24)
 def input_bands(self,u):
  b,t=u.shape[0],u.shape[-1];x=u.reshape(b*2,t)
  normalized,_,_=self.base.nor((x/1e22)*Config.Amp_factor**2)
  return self.base.wav(normalized).reshape(b,2,8,t)
 def forward(self,u,q,mask,bands=None):
  b,t=u.shape[0],u.shape[-1]
  x=u.reshape(b*2,t);qt=q.reshape(b*2,2,16,264)
  m=mask[:,None,:].expand(-1,2,-1).reshape(b*2,t)
  if bands is None:
   p=self.base(x/1e22,qt)*1e22
  else:
   _,mean,std=self.base.nor((x/1e22)*Config.Amp_factor**2)
   features=self.base.seg(bands.reshape(b*2,8,t))
   out=self.base.decoder(self.base.encoder(features,qt)).squeeze(1)
   p=(out*std.unsqueeze(1)+mean.unsqueeze(1))/Config.Amp_factor**2*1e22
  return self.refiner(x,p,m).reshape(b,2,t)
