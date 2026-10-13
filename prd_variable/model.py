import numpy as np,torch
from torch import nn
from prd_lowrank.common import load_backbone
from prd_strategy.latent import WholeWindowAdapter
from .settings import PROBE,RANK,HEAD
class VariableDGF(nn.Module):
    def __init__(self):
        super().__init__(); self.dgf=load_backbone()
        self.adapter=WholeWindowAdapter(np.load(PROBE/'training_direction_basis.npz')['vectors'][:,:RANK])
        self.adapter.coefficient_adapter.load_state_dict(torch.load(HEAD,map_location='cpu',weights_only=True)['state_dict'])
    def input_bands(self,u):return self.dgf.input_bands(u)
    def forward(self,u,q,mask,bands=None,q_low=None):
        p=self.dgf(u,q,mask,bands,q_low)
        return self.adapter(u,p,mask)
