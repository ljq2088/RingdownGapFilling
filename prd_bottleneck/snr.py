"""SNR distribution of the full retained populations and known masks."""
import numpy as np
from prd_physical.common import write
from prd_lowrank.settings import ROOT,CACHE,CAL
from prd_variable.settings import OUT as VAR
from prd_snr.analyze import snr
OUT=ROOT/'FillingGap/results_prd_bottleneck_20261013'

def describe(r):
    q=[0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1]
    return dict(n=len(r),min=float(r.min()),max=float(r.max()),mean=float(r.mean()),quantiles={str(x):float(y) for x,y in zip(q,np.quantile(r,q))},below={str(t):dict(n=int((r<t).sum()),percent=float(100*(r<t).mean())) for t in (5,8,10,15,20,30,50,100)})
def main():
    OUT.mkdir(exist_ok=True);psd=np.load(CAL)['psd'];report={}
    for split,path in [('train',CACHE/'train_truth.npy'),('validation',CACHE/'validation_truth.npy'),('independent_test_fixed66',ROOT/'FillingGap/results_prd_lowrank_20261013/fresh_test/truth.npy')]:
        h=np.load(path,mmap_mode='r');a=[];b=[];c=[];errors=[]
        original=np.load(CACHE/(split+'_mask.npy'),mmap_mode='r') if split in ('train','validation') else np.load(path.parent/'mask.npy',mmap_mode='r')
        variable=np.load(VAR/'cache'/(split+'0')/'mask.npy',mmap_mode='r') if split in ('train','validation') else None
        for lo in range(0,len(h),256):
            hh=h[lo:lo+256].astype(float);r,e=snr(hh,psd);a.append(r);errors.append(e)
            r,e=snr(hh*(1-original[lo:lo+256,None]),psd);b.append(r);errors.append(e)
            if variable is not None:r,e=snr(hh*(1-variable[lo:lo+256,None]),psd);c.append(r);errors.append(e)
        values=dict(full_signal=np.concatenate(a),gated_fixed66=np.concatenate(b))
        if c:values['gated_variable_view0']=np.concatenate(c)
        report[split]={k:describe(r) for k,r in values.items()};report[split]['parseval_max_relative_error']=max(errors)
        np.savez_compressed(OUT/(split+'_snr.npz'),**values)
    write(OUT/'snr_distribution.json',dict(passed=True,definition='joint A/E clean-signal optimal SNR of the528-s crop;4/(fs*N) weighted DFT norm; endpoints excluded; same independently noise-calibrated PSD as mismatch; gated signal SNR changes with gap',range_selected=False,range_origin='physical source prior and detector response; all source draws retained',splits=report))
    for split,v in report.items():print(split,v['full_signal'])
if __name__=='__main__':main()
