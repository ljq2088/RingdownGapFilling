"""Original two-branch DGF with optional zero-initialized A/E residual coupling."""
import torch
from torch import nn
from prd_physical.model import PhysicalDGF
from prd_physical.run import BASE
from .settings import OLD
class JointResidual(nn.Module):
 def __init__(self,width=32):
  super().__init__();self.entry=nn.Conv1d(6,width,7,padding=3)
  self.blocks=nn.ModuleList([nn.Sequential(nn.Conv1d(width,width,7,padding=3*d,dilation=d),nn.GroupNorm(4,width),nn.GELU()) for d in [1,2,4,8,16,32]])
  self.head=nn.Conv1d(width,2,7,padding=3);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
 def forward(self,u,p,mask):
  scale=u.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-4)
  time=torch.linspace(-1,1,u.shape[-1],device=u.device,dtype=u.dtype)[None,None,:].expand(len(u),1,-1)
  f=torch.cat([u/scale,p/scale,mask[:,None,:],time],dim=1)
  y=self.entry(f)
  for block in self.blocks:y=y+.2*block(y)
  return p+.2*scale*self.head(y)
class OptimizedDGF(nn.Module):
 def __init__(self,variant):
  super().__init__();self.variant=variant;self.dgf=PhysicalDGF(BASE)
  self.dgf.load_state_dict(torch.load(OLD/"white/best.pth",map_location="cpu",weights_only=True)["state_dict"])
  self.joint=JointResidual() if variant=="joint_hann" else None
 def input_bands(self,u):return self.dgf.input_bands(u)
 def forward(self,u,q,mask,bands=None):
  p=self.dgf(u,q,mask,bands)
  return p if self.joint is None else self.joint(u,p,mask)
