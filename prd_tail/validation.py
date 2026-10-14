import numpy as np,torch
from scipy.signal import hilbert
from prd_physical.common import Preprocess,summary
from prd_coherent.validation import metrics
from .settings import CACHE,CAL

def tails(h,p):
    z=hilbert(h);w=hilbert(p);amp=abs(z);peak=amp.argmax(-1);a=amp.max(-1,keepdims=True);result={}
    for name,lo,hi in [('intermediate',.01,.1),('weak',.001,.01)]:
        region=(amp>=lo*a)&(amp<hi*a)&(np.arange(1056)[None,None]>=peak[...,None]);den=(h*h*region).sum((1,2));valid=den>0
        error=np.sqrt(((p-h)**2*region).sum((1,2))[valid]/den[valid])
        phase=((np.angle(w*z.conj()))**2*amp**2*region).sum((1,2))/(amp**2*region).sum((1,2)).clip(1e-100)
        result[name]=dict(error=summary(error),phase=summary(np.sqrt(phase[valid])*180/np.pi))
    return result

def evaluate(prediction,truth=None):
    psd=np.load(CAL)['psd'];pre=Preprocess(psd,'white');h=np.load(CACHE/'validation_truth.npy').astype(float) if truth is None else truth
    p=pre.decode(prediction);r=metrics(h,p,psd);r['tail']=tails(h,p);r['score']=r['psd']['mean'];return r

@torch.inference_mode()
def validate(model,data):
    model.eval();out=[]
    for lo in range(0,data.count,128):out.append(model(data.batch(np.arange(lo,min(lo+128,data.count))))[0].cpu().numpy())
    return evaluate(np.concatenate(out))

def eligible(r,b):
    return all(r[k]['mean']<=b[k]['mean']*1.005 for k in ('phase_A','phase_E','snr_ge10')) and r['tail']['intermediate']['error']['mean']<=b['tail']['intermediate']['error']['mean']*1.000001 and r['tail']['intermediate']['phase']['mean']<=b['tail']['intermediate']['phase']['mean']*1.01 and r['tail']['weak']['phase']['mean']<=b['tail']['weak']['phase']['mean']*1.01
