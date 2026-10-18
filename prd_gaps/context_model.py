"""Higher-resolution context models with a noise-filter baseline and gap correction."""
import math
import numpy as np
from scipy.signal import firwin
import torch
from torch import nn
import torch.nn.functional as F
from .model import PartialConv

class SpectralBlock(nn.Module):
    def __init__(self,width,modes=96):
        super().__init__();self.modes=modes
        self.weight=nn.Parameter(torch.randn(width,width,modes,2)*(.02/math.sqrt(width)))
        self.local=nn.Conv1d(width,width,1)
        self.norm=nn.GroupNorm(8,width)
    def forward(self,x):
        # FFT stays in float32 even when convolutions use mixed precision.
        with torch.autocast(device_type=x.device.type,enabled=False):
            spectrum=torch.fft.rfft(x.float());m=min(self.modes,spectrum.shape[-1])
            out=torch.zeros_like(spectrum)
            out[:,:,:m]=torch.einsum('bim,iom->bom',spectrum[:,:,:m],torch.view_as_complex(self.weight[:,:,:m].contiguous()))
            result=torch.fft.irfft(out,n=x.shape[-1])
        return x+F.gelu(self.norm(result.to(x.dtype)+self.local(x)))

class ContextReconstructor(nn.Module):
    def __init__(self,kind='attention',widths=(32,64,128,192),layers=4):
        super().__init__();self.kind=kind
        self.encoder=nn.ModuleList();cin=10
        for width,stride in zip(widths,[4,2,2,2]):
            self.encoder.append(PartialConv(cin,width,stride));cin=width
        if kind=='attention':
            self.context=nn.TransformerEncoder(nn.TransformerEncoderLayer(widths[-1],6,4*widths[-1],dropout=.05,
                        activation='gelu',batch_first=True,norm_first=True),layers,enable_nested_tensor=False)
        elif kind=='spectral':self.context=nn.Sequential(*[SpectralBlock(widths[-1]) for _ in range(layers)])
        else:raise ValueError('kind must be attention or spectral')
        self.decoder=nn.ModuleList();current=widths[-1]
        for width in reversed(widths[:-1]):
            self.decoder.append(nn.Sequential(nn.Conv1d(current+width,width,7,padding=3),nn.GroupNorm(8,width),nn.GELU(),
                                              nn.Conv1d(width,width,5,padding=2),nn.GELU()))
            current=width
        self.out=nn.Sequential(nn.Conv1d(current+10,64,7,padding=3),nn.GELU(),nn.Conv1d(64,2,7,padding=3))
        nn.init.normal_(self.out[-1].weight,std=1e-4);nn.init.zeros_(self.out[-1].bias)
        kernel=firwin(129,.15,fs=2.).astype(np.float32)
        self.register_buffer('baseline_kernel',torch.from_numpy(kernel)[None,None].repeat(2,1,1))

    def forward(self,features):
        valid=features[:,4:6].amax(dim=1,keepdim=True)
        if (valid.sum(dim=-1)==0).any():raise ValueError('Each record needs valid measurements')
        x=features;skips=[]
        for block in self.encoder:x,valid=block(x,valid);skips.append(x)
        if self.kind=='attention':
            tokens=x.transpose(1,2);length,dim=tokens.shape[1:]
            positions=torch.arange(length,device=x.device,dtype=torch.float32)[:,None]
            rates=torch.exp(torch.arange(0,dim,2,device=x.device,dtype=torch.float32)*(-math.log(10000)/dim))
            pos=torch.zeros(length,dim,device=x.device,dtype=torch.float32)
            pos[:,0::2]=torch.sin(positions*rates);pos[:,1::2]=torch.cos(positions*rates)
            x=self.context(tokens+pos.to(tokens.dtype),src_key_padding_mask=~valid[:,0].bool()).transpose(1,2)
        else:x=self.context(x)
        for block,skip in zip(self.decoder,reversed(skips[:-1])):
            x=F.interpolate(x,size=skip.shape[-1],mode='linear',align_corners=False)
            x=block(torch.cat((x,skip),dim=1))
        x=F.interpolate(x,size=features.shape[-1],mode='linear',align_corners=False)
        residual=self.out(torch.cat((x,features),dim=1))
        # Red-noise cancellation in this FIR is sensitive to BF16 rounding.
        with torch.autocast(device_type=features.device.type,enabled=False):
            baseline=F.conv1d(features[:,:2].float(),self.baseline_kernel.float(),padding=64,groups=2)
        return baseline+residual.float()

def focused_loss(prediction,target,valid,info,psd):
    """Phase-sensitive waveform losses, with explicit missing/ROI normalization."""
    pred=prediction.float();target=target.float()
    energy=target.square().mean(dim=(-2,-1)).clamp_min(1e-10)
    loss=.15*torch.log1p((pred-target).square().mean(dim=(-2,-1))/energy).mean()
    terms=[]
    for b in range(len(pred)):
        start,stop=map(int,info[b,:2].tolist())
        p,t=pred[b,:,start:stop],target[b,:,start:stop]
        e=t.square().sum().clamp_min(1e-10)
        time_error=torch.log1p((p-t).square().sum()/e)
        missing=1-valid[b,:,start:stop]
        # Sample average within the gap, with ROI signal energy as a stable scale.
        gap_error=torch.log1p(((p-t).square()*missing).sum()/missing.sum().clamp_min(1)/(e/t.numel()).clamp_min(1e-10))
        local_psd=F.interpolate(psd[None],size=(stop-start)//2+1,mode='linear',align_corners=True)[0]
        floor=local_psd[:,1:].median(dim=-1).values[:,None]*1e-4
        weight=torch.maximum(local_psd,floor).reciprocal().clone();weight[:,0]=0;weight[:,-1]*=.5
        pf,tf=torch.fft.rfft(p),torch.fft.rfft(t)
        pp=(pf.abs().square()*weight).sum();tt=(tf.abs().square()*weight).sum()
        cross=(pf*tf.conj()*weight).real.sum()
        mm=1-cross/torch.sqrt(pp*tt).clamp_min(1e-12)
        spectral_error=torch.log1p(((pf-tf).abs().square()*weight).sum()/tt.clamp_min(1e-10))
        terms.append(time_error+1.5*gap_error+mm+.25*spectral_error)
    return loss+torch.stack(terms).mean()
