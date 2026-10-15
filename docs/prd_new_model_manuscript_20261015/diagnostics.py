import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import json,sys,hashlib
from pathlib import Path
import numpy as np
from scipy.signal import hilbert,periodogram,find_peaks
from pycbc.conversions import get_lm_f0tau
P=Path(__file__).resolve().parent;R=Path('/home/ljq/code/RingdownGapFilling_PRD');sys.path.insert(0,str(R))
from prd_physical.common import Preprocess,metric
from prd_variable.data import variable_mask
F=R/'FillingGap/results_prd_tail_20261015';T=F/'fresh_test';N=1056;fs=2
h=np.load(T/'truth.npy',mmap_mode='r');noise=np.load(T/'noise.npy',mmap_mode='r');pred=np.load(F/'variable_optimized_prediction.npy',mmap_mode='r')
psd=np.load(R/'FillingGap/results_prd_physical_transfer_20261009/calibration_psd.npz')['psd'];pre=Preprocess(psd,'white')
seeds=np.load(T/'seeds.npy');mask=variable_mask(np.load(T/'mask.npy'),seeds,410000000).astype(bool)
M=np.load(F/'per_example_test.npz');snr=M['snr'];mm=M['variable__optimized'];strong=snr>=10
names=['phase_gap','energy','relative','gap_relative','ES','norm_bias','time_mm','coverage','balance','curvature','noise_ratio'];a={k:np.empty(len(h)) for k in names};a['phase_peak']=np.empty((len(h),2));a['inside']=np.empty(len(h),bool);a['extrema']=np.empty((len(h),2),int)
pars=np.load(T/'parameters.npy');tau=np.array([get_lm_f0tau(float(v[9]),float(v[10]),2,2)[1] for v in pars]);sums={k:np.zeros((2,529)) for k in ('target','noise','residual')}
for lo in range(0,len(h),128):
 hi=min(lo+128,len(h));truth=np.asarray(h[lo:hi],float);n=np.asarray(noise[lo:hi],float);p=pre.decode(pred[lo:hi]);m=mask[lo:hi];z=hilbert(truth);w=hilbert(p);delta=np.angle(w*z.conj());amp=abs(z);peak=(amp**2).sum(1).argmax(-1);scale=abs(truth).max((1,2));s=(slice(lo,hi))
 a['inside'][s]=m[np.arange(hi-lo),peak];weight=amp**2*m[:,None];a['phase_gap'][s]=np.sqrt((weight*delta**2).sum((1,2))/weight.sum((1,2)))*180/np.pi
 a['phase_peak'][s]=abs(np.take_along_axis(delta,amp.argmax(-1)[...,None],axis=-1)[...,0])*180/np.pi
 den=(truth**2).sum((1,2));err=(p-truth)**2;a['energy'][s]=(truth**2*m[:,None]).sum((1,2))/den;a['relative'][s]=np.sqrt(err.sum((1,2))/den);a['gap_relative'][s]=np.sqrt((err*m[:,None]).sum((1,2))/(truth**2*m[:,None]).sum((1,2)));a['norm_bias'][s]=np.sqrt((p**2).sum((1,2))/den)-1
 f0=np.fft.rfft(truth);df=np.fft.rfft(p-truth);a['ES'][s]=(abs(df[...,1:-1])**2/psd[:,1:-1]).sum((1,2))/(abs(f0[...,1:-1])**2/psd[:,1:-1]).sum((1,2));a['time_mm'][s]=metric(truth,p)
 assert np.max(abs(metric(truth,p,psd)-mm[s]))<1e-12
 a['coverage'][s]=(527.5-peak*.5)/tau[s];a['balance'][s]=abs(truth).max(-1).min(-1)/scale
 a['curvature'][s]=abs(np.diff(truth,n=2,axis=-1)).max((1,2))/scale
 a['noise_ratio'][s]=abs(truth*(1-m[:,None])+n).max((1,2))/scale
 for j in range(hi-lo):
  for c in range(2):a['extrema'][lo+j,c]=len(find_peaks(abs(truth[j,c,peak[j]:]),prominence=.025*scale[j])[0])
 for k,x in [('target',truth),('noise',n),('residual',p-truth)]:f,v=periodogram(x,fs=fs,window='boxcar',detrend='constant',scaling='density');sums[k]+=v.sum(0)
a['gap_over_tau']=mask.sum(-1)/fs/tau;a['length']=mask.sum(-1)/fs;a['start']=mask.argmax(-1)/fs

def sm(x):return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),p95=float(np.quantile(x,.95)),min=float(x.min()),max=float(x.max()))
out={k:{'all':sm(v),'snr_ge10':sm(v[strong])} for k,v in a.items() if v.ndim==1 and k not in ('inside',)}
out['snr']=sm(snr);out['groups']={k:dict(mismatch=sm(mm[ix]),gap_phase=sm(a['phase_gap'][ix]),gap_relative=sm(a['gap_relative'][ix])) for k,ix in [('inside',a['inside']),('outside',~a['inside']),('inside_strong',a['inside']&strong)]}
out['phase_peak']={c:sm(a['phase_peak'][:,j]) for j,c in enumerate('AE')}
out['energy_bins']=[dict(lo=l,hi=u,mismatch=sm(mm[strong&(a['energy']>=l)&(a['energy']<u)])) for l,u in zip([0,.25,.5,.75],[.25,.5,.75,1.00000001])]
criteria=strong&a['inside']&(a['coverage']>=4)&(a['extrema'].min(-1)>=6)&(a['balance']>.15)&(a['gap_relative']<.07)&(a['relative']<.07)&(a['phase_peak'].max(-1)<3)&(a['noise_ratio']<1.6)&(a['curvature']<.03)
ix=np.flatnonzero(criteria);ix=ix[np.argsort(a['gap_relative'][ix]+a['relative'][ix]+3*a['curvature'][ix])]
out['candidates']=[dict(index=int(i),seed=int(seeds[i]),mismatch=float(mm[i]),snr=float(snr[i]),**{k:v[i].tolist() for k,v in a.items()}) for i in ix[:24]]
np.savez_compressed(P/'publication_metrics.npz',seeds=seeds,snr=snr,mismatch=mm,**a)
np.savez_compressed(P/'population_spectra.npz',frequency=f,**{k:v/len(h) for k,v in sums.items()})
(P/'diagnostics.json').write_text(json.dumps(out,indent=2));print(json.dumps({'groups':out['groups'],'n_candidates':len(ix),'candidates':out['candidates'][:8]},indent=2))
