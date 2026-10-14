import numpy as np,torch
from torch.nn import functional as F
from prd_physical.common import Preprocess
from .settings import CAL
from .common import analytic

class Objective:
    def __init__(self):
        pre=Preprocess(np.load(CAL)['psd'],'white');self.transfer=torch.tensor(pre.asd/pre.sigma.mean(),device='cuda',dtype=torch.float32)
    def __call__(self,p,v,aux,epoch):
        h=v['target'];P=torch.fft.rfft(p);H=torch.fft.rfft(h);a=P[...,1:-1];b=H[...,1:-1]
        tt=b.abs().square().sum((1,2)).clamp_min(1e-12);pp=a.abs().square().sum((1,2)).clamp_min(1e-12)
        mm=1-(a.conj()*b).real.sum((1,2))/(tt*pp).sqrt();rel=(a-b).abs().square().sum((1,2))/tt
        u=torch.fft.irfft(P*self.transfer,n=1056);y=torch.fft.irfft(H*self.transfer,n=1056)
        error=(u-y).square();global_rel=(error.sum((1,2))+2*(error*v['mask'][:,None]).sum((1,2)))/y.square().sum((1,2)).clamp_min(1e-12)
        z=analytic(y);w=analytic(u);amp=z.abs();maxamp=amp.amax(-1,keepdim=True).clamp_min(1e-6)
        peak=amp.square().sum(1).argmax(-1);post=torch.arange(1056,device=p.device)[None,None]>=peak[:,None,None]
        region=(amp>=.01*maxamp)&post
        local=F.smooth_l1_loss((u-y)/(amp+.03*maxamp),torch.zeros_like(y),reduction='none',beta=.5)
        local=(local*region).sum((1,2))/region.sum((1,2)).clamp_min(1)
        denominator=(w.abs().square()+1e-6*amp.square()).sqrt()*amp.clamp_min(1e-6)
        phase=1-(w*z.conj()).real/denominator.clamp_min(1e-12)
        weight=(amp/maxamp).sqrt()*region;phase=(phase*weight).sum((1,2))/weight.sum((1,2)).clamp_min(1e-6)
        rho2=2/1056*tt;reliability=rho2/(rho2+100)
        ramp=min(1.,epoch/10)
        loss=(mm+.2*rel+.02*global_rel+ramp*reliability*(.04*local+.03*phase)).mean()
        if 'parameters' in aux:
            d=aux['parameters']-v['parameters'];loss=loss+.03*(d[:,0].square()+.5*(d[:,1]/.1).square()).mean()
        return loss
