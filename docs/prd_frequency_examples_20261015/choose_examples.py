import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import numpy as np,json
from scipy.signal import hilbert,find_peaks
from pycbc.conversions import get_lm_f0tau
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;R=Path('/home/ljq/code/RingdownGapFilling_PRD');F=R/'FillingGap/results_prd_tail_20261015';T=F/'fresh_test'
a=np.load(P/'publication_metrics.npz');h=np.load(T/'truth.npy',mmap_mode='r');pr=np.load(F/'variable_optimized_prediction.npy',mmap_mode='r');pars=np.load(T/'parameters.npy',mmap_mode='r')
psd=np.load(R/'FillingGap/results_prd_physical_transfer_20261009/calibration_psd.npz')['psd'];asd=np.sqrt(psd);asd[:,0]=asd[:,1];asd[:,-1]=asd[:,-2];t=np.arange(1056)*.5
ix=np.flatnonzero((a['snr']>=10)&(a['mismatch']<.005)&(a['gap_relative']<.15)&(a['relative']<.15)&(a['balance']>.4)&(a['energy']>.08)&(a['noise_ratio']<6))
recs=[]
for i in ix:
 f,tau=get_lm_f0tau(float(pars[i,9]),float(pars[i,10]),2,2)
 v=h[i].astype(float);p=np.fft.irfft(np.fft.rfft(pr[i].astype(float))*asd,n=1056);scale=abs(v).max();env=np.sqrt(np.sum(abs(hilbert(v))**2,axis=0));peak=env.argmax();end=min(527.5,t[peak]+tau*np.log(100));use=(t>=max(0,t[peak]-70))&(t<=end)
 # High-frequency curvature above the local sinusoidal curvature flags sharp onset structure.
 d2=np.diff(v,n=2);expected=(2*np.pi*f*.5)**2
 kink=float(np.max(abs(d2))/(scale*expected));rough=float(np.max(abs(np.diff(p-v,n=2)))/(scale*expected))
 extrema=[len(find_peaks(abs(v[c]),prominence=.03*abs(v[c]).max())[0]) for c in range(2)]
 recs.append(dict(index=int(i),f220=float(f),tau220=float(tau),kink=kink,roughness=rough,extrema=extrema,mismatch=float(a['mismatch'][i]),gap_error=float(a['gap_relative'][i]),energy=float(a['energy'][i]),snr=float(a['snr'][i]),score=kink+rough,peak_s=float(t[peak]),display_end_s=float(end)))
recs.sort(key=lambda r:r['score']);(P/'candidates.json').write_text(json.dumps(recs,indent=2))
for label,pool in [('smooth',recs),('higher_frequency',[r for r in recs if r['f220']>.025])]:
 cand=pool[:12]
 if not cand:continue
 fig,axes=plt.subplots(len(cand),2,figsize=(13,2.1*len(cand)),squeeze=False)
 for j,r in enumerate(cand):
  i=r['index'];v=h[i];p=np.fft.irfft(np.fft.rfft(pr[i].astype(float))*asd,n=1056);scale=abs(v).max()
  for c in range(2):
   ax=axes[j,c];ax.plot(t,v[c]/scale,color='#183b60',lw=1.2);ax.plot(t,p[c]/scale,color='#ca6023',ls='--',lw=1);ax.axvspan(a['start'][i],a['start'][i]+a['length'][i],alpha=.15);ax.set(xlim=(0,min(400,r['display_end_s'])),title=f'{i} {"AE"[c]} f={r["f220"]:.3f} M={r["mismatch"]:.4f} gap={r["gap_error"]:.3f} kink={r["kink"]:.1f} extrema={r["extrema"]}');ax.grid(alpha=.15)
 fig.tight_layout();fig.savefig(P/(label+'.png'),dpi=100);plt.close(fig)
print('eligible',len(recs));print('high-frequency',json.dumps([r for r in recs if r['f220']>.025][:12],indent=2));print('smooth',json.dumps(recs[:6],indent=2))
