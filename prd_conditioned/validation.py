import numpy as np,torch
from prd_physical.common import Preprocess,metric,summary,N,FS
from .settings import CAL,CACHE,VAL_N
from .model import reconstruct

def validate(model,data,basis=None,live=False):
    model.eval();cal=np.load(CAL);pre=Preprocess(cal['psd'],'white')
    truth=np.load(CACHE/'validation_truth.npy').astype(float);H=np.fft.rfft(truth)
    snr=np.sqrt((4/(FS*N)*((abs(H[...,1:-1])**2)/cal['psd'][:,1:-1])).sum((1,2)))
    sets={}
    with torch.inference_mode():
        for j,name in enumerate(('variable','fixed66','fixed100')):
            out=[]
            for lo in range(0,VAL_N,128 if live else 512):
                hi=min(lo+(128 if live else 512),VAL_N);v=data.batch(np.arange(lo,hi)+j*VAL_N)
                y=model(v['input'],v['q'],v['mask'],v['bands'],v['q_low']) if live else reconstruct(model,v,basis)
                out.append(y.cpu().numpy())
            p=pre.decode(np.concatenate(out));m=metric(truth,p,cal['psd']);sets[name]=dict(psd=summary(m),time=summary(metric(truth,p)),snr_ge10=summary(m[snr>=10]))
    return dict(score=sets['variable']['psd']['mean'],sets=sets)
