import numpy as np,torch
from scipy.signal import hilbert
from prd_physical.common import N,FS,Preprocess,metric,summary
from .settings import CAL,CACHE,VAL_N

def phase(h,p):
    z=hilbert(h,axis=-1);w=hilbert(p,axis=-1);j=abs(z).argmax(-1)
    i=np.arange(len(h))[:,None];c=np.arange(2)[None,:]
    return abs(np.angle(w[i,c,j]*z[i,c,j].conj()))*180/np.pi

def metrics(h,p,psd):
    f=np.fft.rfft(h)[...,1:-1];rho=np.sqrt((4/(FS*N)*abs(f)**2/psd[:,1:-1]).sum((1,2)))
    mm=metric(h,p,psd);pe=phase(h,p)
    return dict(psd=summary(mm),snr_ge10=summary(mm[rho>=10]),phase_A=summary(pe[:,0]),phase_E=summary(pe[:,1]),phase_mean=float(pe.mean()),time=summary(metric(h,p)))

def validate(model,data,batch=128):
    model.eval();psd=np.load(CAL)['psd'];pre=Preprocess(psd,'white');h=np.load(CACHE/'validation_truth.npy').astype(float);sets={}
    with torch.inference_mode():
        for j,name in enumerate(('variable','fixed66','fixed100')):
            out=[]
            for lo in range(0,VAL_N,batch):
                v=data.batch(np.arange(lo,min(lo+batch,VAL_N))+j*VAL_N)
                out.append(model(v['input'],v['q'],v['mask'],v['bands'],v['q_low']).cpu().numpy())
            sets[name]=metrics(h,pre.decode(np.concatenate(out)),psd)
    return dict(score=sets['variable']['psd']['mean'],sets=sets)

def eligible(result,baseline):
    # Preserve both channel phases and the high-SNR population while reducing
    # the prespecified full-population primary metric.
    a=result['sets']['variable'];b=baseline['sets']['variable']
    return all(a[k]['mean']<=b[k]['mean']*1.005 for k in ('phase_A','phase_E','snr_ge10'))
