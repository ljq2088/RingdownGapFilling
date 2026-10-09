"""Original DGF plus its small mask residual adapter, in calibrated working units."""
import torch
from torch import nn
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
 def forward(self,u,q,mask):
  b,t=u.shape[0],u.shape[-1]
  x=u.reshape(b*2,t);qt=q.reshape(b*2,2,16,264)
  m=mask[:,None,:].expand(-1,2,-1).reshape(b*2,t)
  p=self.base(x/1e22,qt)*1e22
  return self.refiner(x,p,m).reshape(b,2,t)
