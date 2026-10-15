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
ix=np.flatnonzero((a['mismatch']<.002)&(a['relative']<.12)&(a['gap_relative']<.12)&(a['coverage']>3)&(a['extrema'].min(1)>=6)&(a['balance']>.35)&(a['noise_ratio']<3)&(a['energy']>.12))
records=[]
for i in ix:
 v=np.asarray(h[i],float);p=np.fft.irfft(np.fft.rfft(pred[i].astype(float))*asd,n=1056);scale=abs(v).max()
 # Curvature concentration distinguishes isolated onset kinks from resolved oscillations.
 d2=np.diff(v,n=2);dp2=np.diff(p,n=2);smooth=np.max(abs(d2))/np.sqrt(np.mean(d2*d2));rough=np.max(abs(dp2-d2))/scale
 records.append(dict(index=int(i),curv_concentration=float(smooth),prediction_roughness=float(rough),score=float(smooth+100*rough+20*a['relative'][i]),mismatch=float(a['mismatch'][i]),inside=bool(a['inside'][i])))
records=sorted(records,key=lambda x:x['score']);(P/'waveform_candidates.json').write_text(json.dumps(records,indent=2))
for batch in range(3):
 cand=records[batch*16:(batch+1)*16];fig,axes=plt.subplots(len(cand),2,figsize=(14,2*len(cand)))
 for row,r in enumerate(cand):
  i=r['index'];v=h[i];p=np.fft.irfft(np.fft.rfft(pred[i].astype(float))*asd,n=1056);scale=abs(v).max()
  for c in range(2):
   ax=axes[row,c];ax.plot(t,v[c]/scale,color='#163759',lw=1);ax.plot(t,p[c]/scale,color='#cb6320',lw=.9,ls='--');ax.axvspan(a['start'][i],a['start'][i]+a['length'][i],alpha=.15);ax.set(xlim=(0,528),ylim=(-1.1,1.1),title=f'{i} {"AE"[c]} M={a["mismatch"][i]:.4g} gaperr={a["gap_relative"][i]:.3f} score={r["score"]:.2f} inside={r["inside"]}');ax.grid(alpha=.1)
 fig.tight_layout();fig.savefig(P/f'candidates_{batch}.png',dpi=90);plt.close(fig)
print(len(records));print(records[:8])
