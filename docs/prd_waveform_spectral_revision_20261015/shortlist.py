import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import numpy as np,json
from scipy.signal import hilbert,find_peaks,periodogram
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;R=Path('/home/ljq/code/RingdownGapFilling_PRD');F=R/'FillingGap/results_prd_tail_20261015';T=F/'fresh_test'
a=np.load(P/'publication_metrics.npz');h=np.load(T/'truth.npy',mmap_mode='r');pred=np.load(F/'variable_optimized_prediction.npy',mmap_mode='r');noise=np.load(T/'noise.npy',mmap_mode='r')
psd=np.load(R/'FillingGap/results_prd_physical_transfer_20261009/calibration_psd.npz')['psd'];asd=np.sqrt(psd);asd[:,0]=asd[:,1];asd[:,-1]=asd[:,-2]
t=np.arange(1056)*.5
chosen=[2428,4409,4061,4884,1593,5153]
fig,axes=plt.subplots(len(chosen),2,figsize=(14,3*len(chosen)))
for row,i in enumerate(chosen):
 v=np.asarray(h[i],float);p=np.fft.irfft(np.fft.rfft(pred[i].astype(float))*asd,n=1056);scale=abs(v).max();m=(t>=a['start'][i])&(t<a['start'][i]+a['length'][i]);d=v*(1-m)+noise[i]
 print(json.dumps(dict(index=i,**{k:a[k][i].tolist() for k in a.files})))
 for c in range(2):
  ax=axes[row,c];ax.plot(t,d[c]/scale,color='#bbc3cd',lw=.4);ax.plot(t,v[c]/scale,color='#163759',lw=1.3);ax.plot(t,p[c]/scale,color='#cb6320',lw=1.2,ls='--');ax.axvspan(a['start'][i],a['start'][i]+a['length'][i],alpha=.15);ax.set(xlim=(0,350),title=f'{i} {"AE"[c]}');ax.grid(alpha=.1)
fig.tight_layout();fig.savefig(P/'shortlist.png',dpi=130)
