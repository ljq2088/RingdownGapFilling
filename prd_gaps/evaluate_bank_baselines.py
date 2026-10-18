"""Frozen-source comparisons for simple baselines and optional pilot weights."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.signal import firwin
import torch
from torch import nn
from torch.utils.data import DataLoader
from .online import SourceBank
from .train_context import validation

class Baseline(nn.Module):
    def __init__(self,lowpass):
        super().__init__();self.lowpass=lowpass
        self.register_buffer('kernel',torch.tensor(firwin(129,.15,fs=2.),dtype=torch.float32)[None,None].repeat(2,1,1))
    def forward(self,x):
        return torch.nn.functional.conv1d(x[:,:2],self.kernel,padding=64,groups=2) if self.lowpass else x[:,:2]

def run(data):
    torch.set_num_threads(4);dataset=SourceBank(data,'val');loader=DataLoader(dataset,batch_size=8,num_workers=2)
    result={}
    for name,lowpass in [('zero_filled_observation',False),('lowpass_015hz',True)]:
        result[name]=validation(Baseline(lowpass),loader,dataset.psd,'cpu',False)
        print(json.dumps({name:{k:v for k,v in result[name].items() if k!='samples'}}),flush=True)
    (Path(__file__).parent/'reports/context_bank_baselines.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);run(p.parse_args().data)
