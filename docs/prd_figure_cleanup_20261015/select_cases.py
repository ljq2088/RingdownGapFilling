from pathlib import Path
import sys,json,numpy as np
from scipy.signal import hilbert,find_peaks
ROOT=Path('/home/ljq/code/RingdownGapFilling_PRD');sys.path.insert(0,str(ROOT))
from prd_conditioned.settings import OUT,CAL
from prd_physical.common import Preprocess
from prd_variable.data import variable_mask
HERE=Path(__file__).resolve().parent
h=np.load(OUT/'fresh_test/truth.npy').astype(float);p=Preprocess(np.load(CAL)['psd'],'white').decode(np.load(OUT/'variable_optimized_prediction.npy'))
seeds=np.load(OUT/'fresh_test/seeds.npy');m=variable_mask(np.load(OUT/'fresh_test/mask.npy'),seeds,410000000)
rows=json.loads((HERE.parent/'optimization_round_20261015/oscillatory_candidates.json').read_text())['candidates']
zh=hilbert(h,axis=-1);zp=hilbert(p,axis=-1);t=np.arange(1056)*.5
for r in rows:
 i=r['index'];tails=[];phases=[];kinks=[];early=[]
 for c in range(2):
  env=abs(zh[i,c]);peak=env.argmax();region=(np.arange(1056)>peak)&(env>=.01*env.max())&(env<=.1*env.max())
  tails.append(float(np.sqrt(np.sum((p[i,c,region]-h[i,c,region])**2)/max(np.sum(h[i,c,region]**2),1e-99))))
  phases.append(float(np.sqrt(np.average(np.angle(zp[i,c,region]*zh[i,c,region].conj())**2,weights=env[region]**2))*180/np.pi) if region.any() else 999.)
  kinks.append(float(np.max(abs(np.diff(h[i,c],n=2)))/max(abs(h[i,c]).max(),1e-99)))
  early.append(float(np.sqrt(np.mean((p[i,c,:60]-h[i,c,:60])**2))/max(abs(h[i,c]).max(),1e-99)))
 r.update(tail_relative= max(tails),tail_phase=max(phases),native_curvature=max(kinks),initial_residual=max(early))
 # Good global/gap reconstruction, visible decaying oscillations in both channels.
 r['qualified']=bool(r['relative_error']<=.08 and r['gap_relative_error']<=.08 and max(r['phase_A'],r['phase_E'])<2 and min(r['visible_extrema'])>=6)
 r['display_score']=float(r['tail_relative']+r['native_curvature']*4+r['initial_residual']*10+r['relative_error']+r['gap_relative_error'])
rows.sort(key=lambda r:r['display_score'])
(HERE/'candidate_metrics.json').write_text(json.dumps(rows,indent=2))
qualified=[r for r in rows if r['qualified']]
print(json.dumps(qualified[:18],indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(8,2,figsize=(14,22),layout='constrained')
for ax,r in zip(axes.flat,qualified[:16]):
 i=r['index'];scale=abs(h[i]).max();st=m[i].argmax()/2;end=st+m[i].sum()/2
 ax.axvspan(st,end,color='#e6edf5')
 for c,col in enumerate(['#203a59','#14847b']):
  ax.plot(t,h[i,c]/scale,color=col,lw=1);ax.plot(t,p[i,c]/scale,color=col,ls='--',lw=.8)
 ax.set(xlim=(0,528),title=f'{i}: {r["gap_seconds"]:.0f}s; MM {r["mismatch"]:.4g}; tail {r["tail_relative"]:.2f}; kink {r["native_curvature"]:.3g}',ylim=(-1.1,1.1));ax.grid(alpha=.15)
fig.savefig(HERE/'candidates.png',dpi=130);plt.close(fig)
np.savez_compressed(HERE/'candidate_samples.npz',time=t,index=np.array([r['index'] for r in qualified]),seeds=seeds[[r['index'] for r in qualified]],truth=h[[r['index'] for r in qualified]],reconstruction=p[[r['index'] for r in qualified]],mask=m[[r['index'] for r in qualified]])
