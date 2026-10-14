"""Illustrative best cases in the pre-existing clean-envelope-peak-in-gap subgroup.

This selection modifies display examples only. All 1000 sources remain in every
previous population statistic. Stored inference outputs are reused verbatim.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
DATA=Path('/home/ljq/code/RingdownGapFilling/FillingGap/results_augmented_longrun_20261008')
METRICS=Path('/home/ljq/code/RingdownGapFilling_PRD/docs/prd_revision_20261009/evaluation/per_example_metrics.npz')

with np.load(METRICS) as z:
    metric={k:z[k] for k in z.files}
eligible=np.flatnonzero(metric['peak_in_gap'])
selected=eligible[np.argsort(metric['augmented__psd_mismatch'][eligible],kind='stable')[:3]]
with np.load(DATA/'final_test_baseline.npz') as z:
    h=z['truth'][selected].astype(np.float64)
    mask=z['mask'][selected]
    starts=z['gap_starts'][selected]
with np.load(DATA/'final_test_predictions.npz') as z:
    original=z['baseline'][selected].astype(np.float64)
    augmented=z['augmented'][selected].astype(np.float64)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':8.5,
                     'legend.fontsize':8,'pdf.fonttype':42,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(3,1,figsize=(7.35,5.30),constrained_layout=True)
t=np.arange(1056)/2
records=[]
for rank,(ax,idx,target,base,pred,m,st) in enumerate(zip(axes,selected,h,original,augmented,mask,starts),1):
    scale=np.max(np.abs(target))
    assert scale>0
    energy=float(np.sum(target**2*m)/np.sum(target**2))
    ax.axvspan(st/2,(st+66)/2,color='#dfe7f0',alpha=.7,lw=0,zorder=0)
    ax.plot(t,target/scale,color='#263348',lw=1.4,label='Clean target',zorder=3)
    ax.plot(t,base/scale,color='#96a2b1',lw=1.1,ls=':',label='Original DGF',zorder=2)
    ax.plot(t,pred/scale,color='#c66a22',lw=1.1,ls='--',label='Augmented refinement',zorder=4)
    mm=float(metric['augmented__psd_mismatch'][idx])
    seed=int(metric['seeds'][idx])
    ax.set_title(f'({chr(96+rank)}) Seed {seed}  |  Direct PSD mismatch = {mm:.3g}',
                 fontsize=9,loc='left',color='#263348',pad=6)
    ax.set_xlim(max(0,(st-100)/2),min(528,(st+210)/2))
    ax.set_ylim(-1.14,1.14)
    ax.set_ylabel(r'Amplitude / $\max |h|$')
    ax.grid(axis='y',alpha=.16,lw=.6)
    ax.tick_params(labelsize=8)
    records.append({'rank_in_peak_in_gap_subgroup':rank,'index':int(idx),'seed':seed,
        'direct_psd_mismatch':mm,'peak_abs_phase_degrees':float(metric['augmented__peak_abs_phase_deg'][idx]),
        'masked_signal_squared_norm_fraction':energy,'gap_seconds':[float(st/2),float((st+66)/2)],
        'shared_display_amplitude_divisor':float(scale)})
axes[0].legend(loc='upper right',frameon=True,facecolor='white',edgecolor='#d6dfe8',framealpha=.96)
axes[-1].set_xlabel('Time from segment start (s)')
for ext in ['pdf','svg','png']:
    fig.savefig(HERE/'source_edited'/('examples_best_peak.'+ext),dpi=200,facecolor='white')
plt.close(fig)
(HERE/'best_examples_selection.json').write_text(json.dumps({
    'population':'Earlier benchmark: 1000 independent test sources',
    'eligible_subgroup':'Clean analytic-envelope peak inside the 33-s mask',
    'eligible_count':len(eligible),'selection':'Three smallest augmented direct PSD mismatches within the subgroup',
    'purpose':'Illustrative successful signal recovery in an informative gap',
    'population_statistics_changed':False,'time_phase_alignment_applied':False,
    'plot_normalization':'Same target peak amplitude divides all three plotted waveforms per panel',
    'inference_outputs_modified':False,'selected':records},indent=2))
print(json.dumps(records,indent=2))
