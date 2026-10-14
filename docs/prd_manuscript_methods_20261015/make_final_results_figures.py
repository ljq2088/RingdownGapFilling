"""Present existing benchmark advantages with complete population statistics.

Empirical distributions and periodogram means use every stored test source.
The individual phase illustration reuses Fig. 4(b)'s disclosed successful case.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert, periodogram

HERE=Path(__file__).resolve().parent
OUT=HERE/'source_edited'
ROOT=Path('/home/ljq/code/RingdownGapFilling')
RUN=ROOT/'FillingGap/results_augmented_longrun_20261008'
METRICS=Path('/home/ljq/code/RingdownGapFilling_PRD/docs/prd_revision_20261009/evaluation/per_example_metrics.npz')
with np.load(METRICS) as z: metric={k:z[k] for k in z.files}
with np.load(RUN/'final_test_baseline.npz') as z:
    h=z['truth'].astype(float); starts=z['gap_starts']; mask=z['mask']
with np.load(RUN/'final_test_raw.npz') as z: noise=z['noise'].astype(float)
with np.load(RUN/'final_test_predictions.npz') as z: pred={k:z[k].astype(float) for k in z.files}
assert h.shape==(1000,1056)
COLORS={'augmented':'#c66a22','interpolation':'#29988b'}
LABELS={'baseline':'Original DGF','previous':'First refinement','augmented':'DGF','interpolation':'Linear interpolation'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,
                     'legend.fontsize':7.6,'pdf.fonttype':42,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False})

def save(fig,name):
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/(name+'.'+ext),dpi=200,facecolor='white')
    plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(7.35,3.05))
fig.subplots_adjust(left=.08,right=.99,top=.82,bottom=.26,wspace=.29)
means={}
for j,key in enumerate(['time_mismatch','psd_mismatch']):
    ax=axes[j]
    for name in COLORS:
        values=metric[name+'__'+key]
        assert values.shape==(1000,) and np.isfinite(values).all()
        ax.step(np.sort(values),np.arange(1,1001)/1000,where='post',color=COLORS[name],
                label=LABELS[name],lw=1.65 if name=='augmented' else 1.20)
    ax.set(xscale='log',xlim=(min(1e-4,.8*min(metric[n+'__'+key].min() for n in COLORS)),2),ylim=(0,1.015))
    ax.set_ylabel('Fraction of all 1000 test sources')
    ax.set_xlabel('Direct time-domain mismatch' if j==0 else 'Direct PSD-weighted mismatch')
    old=float(metric['baseline__'+key].mean()); new=float(metric['augmented__'+key].mean())
    means[key]={'original_mean':old,'augmented_mean':new,'relative_mean_reduction':1-new/old}
    heading='(a) Waveform agreement' if j==0 else '(b) Noise-weighted agreement'
    ax.set_title(heading+f'\nDGF mean mismatch: {new:.5f}',loc='left',fontsize=8.6,color='#263348',pad=8)
    ax.grid(alpha=.14,lw=.6)
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.015))
save(fig,'mismatch_cdf_presented')

fig,ax=plt.subplots(figsize=(3.55,2.85),constrained_layout=True)
f,pn=periodogram(noise,fs=2,axis=1)
_,po=periodogram(pred['baseline']-h,fs=2,axis=1)
_,pa=periodogram(pred['augmented']-h,fs=2,axis=1)
f=f[1:-1]; pn=pn.mean(0)[1:-1]; po=po.mean(0)[1:-1]; pa=pa.mean(0)[1:-1]


ax.loglog(f,pa,color=COLORS['augmented'],lw=1.55,label='DGF waveform residual')
ax.loglog(f,pn,color='#263348',ls='--',lw=.95,label='Simulated noise')
ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Mean periodogram\n(benchmark units² / Hz)')
ax.grid(alpha=.14,lw=.6); ax.legend(loc='upper right',frameon=False,fontsize=7)
save(fig,'error_psd_presented')

selection=json.loads((HERE/'best_examples_selection.json').read_text())
i=selection['selected'][1]['index']
z=hilbert(h[i]); peak=int(np.argmax(np.abs(z)))
phase_curves={name:np.angle(hilbert(pred[name][i])*z.conj())*180/np.pi for name in ['augmented']}
t=np.arange(1056)/2
phase_display=np.abs(z)>.1*np.abs(z).max()
fig,axes=plt.subplots(1,2,figsize=(7.35,3.1))
fig.subplots_adjust(left=.08,right=.99,top=.82,bottom=.27,wspace=.35)
ax=axes[0]
for name in ['augmented']:
    values=metric[name+'__peak_abs_phase_deg']
    ax.step(np.sort(values),np.arange(1,1001)/1000,where='post',color=COLORS[name],
            label=LABELS[name],lw=1.65 if name=='augmented' else 1.2)
ax.set(xscale='symlog',xlim=(0,180),ylim=(0,1.015)); ax.set_xscale('symlog',linthresh=.2)
ax.set_xlabel('Absolute peak-phase deviation (deg)')
ax.set_ylabel('Fraction of all 1000 test sources')
old=float(metric['baseline__peak_abs_phase_deg'].mean()); new=float(metric['augmented__peak_abs_phase_deg'].mean())
ax.set_title(f'(a) Full-population phase accuracy\nDGF mean absolute error: {new:.2f}°',loc='left',fontsize=8.6,color='#263348',pad=8)
ax.grid(alpha=.14,lw=.6)
ax=axes[1]; st=starts[i]
ax.axvspan(st/2,(st+66)/2,color='#dfe7f0',alpha=.7,lw=0)
target_phase=np.unwrap(np.angle(z))
ax.plot(t,np.where(phase_display,target_phase,np.nan),color='#263348',lw=1.45,label='Clean target')
for name in phase_curves:
    curve=np.where(phase_display,target_phase+phase_curves[name]*np.pi/180,np.nan)
    ax.plot(t,curve,color=COLORS[name],lw=1.15,ls='--' if name=='augmented' else ':',label=LABELS[name])
ax.axvline(peak/2,color='#718096',lw=.75,ls=':')
ax.set_xlim((st-50)/2,(st+130)/2)
visible=phase_display & (t>=(st-50)/2) & (t<=(st+130)/2)
low=target_phase[visible].min(); high=target_phase[visible].max(); margin=.10*(high-low)
ax.set_ylim(low-margin,high+margin)
ax.set_xlabel('Time from segment start (s)'); ax.set_ylabel('Composite analytic phase (rad)')
ax.set_title(f'(b) Successful case: seed {int(metric["seeds"][i])}\nPeak error: {metric["augmented__peak_abs_phase_deg"][i]:.3f}°',
            loc='left',fontsize=8.6,color='#263348',pad=8)
ax.grid(axis='y',alpha=.14,lw=.6)
handles,labels=axes[0].get_legend_handles_labels()
clean_handle=axes[1].get_legend_handles_labels()[0][0]
fig.legend([clean_handle]+handles,['Clean target']+labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.01))
save(fig,'phase_presented')
(HERE/'results_presentation_checks.json').write_text(json.dumps({
    'population_count':1000,'population_rows_selected_for_statistics':'all',
    'figure_scope':'Final DGF estimator and interpolation reference; no checkpoint progression', 'means':means,'phase_original_mean_degrees':old,'phase_augmented_mean_degrees':new,
    'phase_example_seed':int(metric['seeds'][i]),'phase_example_source':'Fig. 4(b)',
    'phase_display_condition':'Clean analytic-envelope amplitude exceeds 10% of its peak',
    'phase_time_or_phase_alignment':False,
    'phase_unwrapping':'Clean-target branch plus wrapped prediction-target phase difference; phase modulo 2pi unchanged',
    'spectrum_aggregation':'Arithmetic mean of all 1000 one-sided error/noise periodograms',
    'metric_arrays_modified':False},indent=2))
print(json.dumps(means,indent=2))
