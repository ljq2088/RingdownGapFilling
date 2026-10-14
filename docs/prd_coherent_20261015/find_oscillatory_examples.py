from pathlib import Path
import json,sys,numpy as np
from scipy.signal import hilbert,find_peaks
ROOT=Path('/home/ljq/code/RingdownGapFilling_PRD');sys.path.insert(0,str(ROOT))
from prd_conditioned.settings import OUT,CAL
from prd_physical.common import Preprocess,metric
from prd_variable.data import variable_mask
from pycbc.conversions import get_lm_f0tau
HERE=Path(__file__).resolve().parent
p=np.load(OUT/'fresh_test/parameters.npy');h=np.load(OUT/'fresh_test/truth.npy').astype(float);noise=np.load(OUT/'fresh_test/noise.npy',mmap_mode='r');seeds=np.load(OUT/'fresh_test/seeds.npy');m=variable_mask(np.load(OUT/'fresh_test/mask.npy'),seeds,410000000)
pre=Preprocess(np.load(CAL)['psd'],'white');pred=pre.decode(np.load(OUT/'variable_optimized_prediction.npy'));stats=np.load(OUT/'per_example_test.npz');mm=stats['variable__optimized'];snr=stats['snr'];zh=hilbert(h,axis=-1);zp=hilbert(pred,axis=-1)
peak=np.sqrt((abs(zh)**2).sum(1)).argmax(-1);peaks=abs(zh).argmax(-1);pe=abs(np.angle(zp[np.arange(len(h))[:,None],np.arange(2)[None],peaks]*zh[np.arange(len(h))[:,None],np.arange(2)[None],peaks].conj()))*180/np.pi
f=np.array([get_lm_f0tau(x[9],x[10],2,2)[0] for x in p]);tau=np.array([get_lm_f0tau(x[9],x[10],2,2)[1] for x in p]);energy=(h*h).sum((1,2));gapenergy=(h*h*m[:,None]).sum((1,2));balance=(h*h).sum(-1).min(-1)/(h*h).sum(-1).max(-1);rel=np.sqrt(((pred-h)**2).sum((1,2))/energy);gaprel=np.sqrt(((pred-h)**2*m[:,None]).sum((1,2))/gapenergy)
eligible=(m[np.arange(len(h)),peak]>0)&(528-peak/2>=4*tau)&(snr>=10)&(balance>=.1)
candidates=np.flatnonzero(eligible);candidates=candidates[np.argsort(mm[candidates],kind='stable')]
rows=[]
for i in candidates:
    n_extrema=[]
    for c in range(2):
        local=h[i,c];a=abs(local).max();n_extrema.append(int(len(find_peaks(abs(local),height=a*.05,prominence=a*.025,distance=max(2,int(.2/f[i])))[0])))
    rows.append(dict(index=int(i),seed=int(seeds[i]),mismatch=float(mm[i]),frequency_220=float(f[i]),tau_220=float(tau[i]),post_peak_damping_times=float((528-peak[i]/2)/tau[i]),snr=float(snr[i]),relative_error=float(rel[i]),gap_relative_error=float(gaprel[i]),gap_energy_fraction=float(gapenergy[i]/energy[i]),phase_A=float(pe[i,0]),phase_E=float(pe[i,1]),visible_extrema=n_extrema,balance=float(balance[i]),gap_seconds=float(m[i].sum()/2)))
report=dict(cohort_count=len(candidates),selection_conditions=dict(peak_inside_gap=True,post_detector_peak_damping_times_at_least=4,network_snr_at_least=10,minimum_channel_energy_ratio=.1),candidates=rows)
(HERE/'oscillatory_candidates.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(cohort_count=len(candidates),top20=rows[:20]),indent=2))
qualified=[r for r in rows if max(r['phase_A'],r['phase_E'])<=1.5 and r['relative_error']<=.08 and r['gap_relative_error']<=.08 and min(r['visible_extrema'])>=6]
selected=[min([r for r in qualified if lo<=r['gap_seconds']<=hi],key=lambda r:r['mismatch']) for lo,hi in [(32,38),(39,45),(46,52)]]
selection=dict(physical_cohort_count=len(candidates),qualified_count=len(qualified),physical_conditions=report['selection_conditions'],illustration_quality_conditions=dict(maximum_channel_peak_phase_degrees=1.5,maximum_global_relative_error=.08,maximum_gap_relative_error=.08,minimum_visible_extrema_each_channel=6),duration_bins_seconds=[[32,38],[39,45],[46,52]],ranking='lowest joint direct PSD mismatch in each duration bin among qualified successful examples',selected=selected,all_population_statistics_unchanged=True)
(HERE/'selected_oscillatory_examples.json').write_text(json.dumps(selection,indent=2));print(json.dumps(selection,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(6,3,figsize=(16,15),layout='constrained');t=np.arange(1056)/2
for ax,r in zip(axes.flat,rows[:18]):
    i=r['index'];scale=abs(h[i]).max();st=m[i].argmax()/2;end=st+m[i].sum()/2
    ax.axvspan(st,end,color='#dfe7f0');ax.plot(t,h[i,0]/scale,color='#203a59',lw=1);ax.plot(t,pred[i,0]/scale,color='#ce6b25',ls='--',lw=.9);ax.plot(t,h[i,1]/scale,color='#14847b',lw=.85,alpha=.8)
    ax.set(xlim=(max(0,st-35),min(528,st+210)),title=f'{i} M={mm[i]:.5f} | gapErr={gaprel[i]:.3f}\nphase={pe[i,0]:.1f}/{pe[i,1]:.1f} | ext={r["visible_extrema"]}');ax.grid(alpha=.15)
fig.savefig(HERE/'oscillatory_candidates.png',dpi=135);plt.close(fig)
