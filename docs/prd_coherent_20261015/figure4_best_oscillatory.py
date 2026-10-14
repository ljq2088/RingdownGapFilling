from pathlib import Path
import sys,json,zipfile,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE=Path(__file__).resolve().parent
DEST=HERE.parent/'figure4_oscillatory_20261015';DEST.mkdir(exist_ok=True)
ROOT=Path('/home/ljq/code/RingdownGapFilling_PRD');sys.path.insert(0,str(ROOT))
from prd_conditioned.settings import OUT as RUN,CAL
from prd_physical.common import Preprocess,sha,metric
from prd_variable.data import variable_mask
source=DEST/'source_edited';source.mkdir(exist_ok=True)
with zipfile.ZipFile(DEST/'overleaf_before.zip') as z:z.extractall(source)
report=json.loads((HERE.parent/'manuscript_unified_20261015/latest_test_summary.json').read_text())
assert sha(RUN/'finetune/best.pth')==report['checkpoint_sha256']
selection=json.loads((HERE/'selected_oscillatory_examples.json').read_text())
chosen=np.array([r['index'] for r in selection['selected']]);seeds=np.load(RUN/'fresh_test/seeds.npy');masks=variable_mask(np.load(RUN/'fresh_test/mask.npy'),seeds,410000000)
truth=np.load(RUN/'fresh_test/truth.npy',mmap_mode='r')[chosen].astype(float)
noise=np.load(RUN/'fresh_test/noise.npy',mmap_mode='r')[chosen].astype(float)
mask=masks[chosen];pre=Preprocess(np.load(CAL)['psd'],'white')
pred=pre.decode(np.load(RUN/'variable_optimized_prediction.npy',mmap_mode='r')[chosen]);gapped=truth*(1-mask[:,None]);observed=gapped+noise;t=np.arange(1056)/2
mm=metric(truth,pred,np.load(CAL)['psd']);assert abs(mm-np.array([r['mismatch'] for r in selection['selected']])).max()<1e-12
np.savez_compressed(DEST/'figure4_native_samples.npz',time=t,index=chosen,seeds=seeds[chosen],truth=truth,noise=noise,mask=mask,gapped_truth=gapped,observation=observed,reconstruction=pred)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'legend.fontsize':7.5,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'path.simplify':False})
colors=dict(noise='#aab3bf',truth='#1e334f',gap='#14847b',pred='#cc5f20',shade='#e6edf5')
fig,axes=plt.subplots(3,2,figsize=(7.35,7.25),layout='constrained')
fig.set_constrained_layout_pads(w_pad=.035,h_pad=.04,hspace=.09,wspace=.05)
for row,i in enumerate(chosen):
    st=mask[row].argmax()/2;end=st+mask[row].sum()/2;scale=abs(truth[row]).max()
    for c in range(2):
        ax=axes[row,c];ax.axvspan(st,end,color=colors['shade'],lw=0,zorder=0)
        ax.plot(t,observed[row,c]/scale,color=colors['noise'],lw=.5,alpha=.75,zorder=1)
        ax.plot(t,truth[row,c]/scale,color=colors['truth'],lw=1.5,zorder=3)
        ax.plot(t,gapped[row,c]/scale,color=colors['gap'],lw=1.1,ls=(0,(4,2,1,2)),zorder=4)
        ax.plot(t,pred[row,c]/scale,color=colors['pred'],lw=1.05,ls=(0,(3,2)),zorder=5)
        ylim=max(1.18,float(abs(observed[row,c]/scale).max())*1.03)
        ax.set(xlim=(0,528),ylim=(-ylim,ylim),xticks=[0,100,200,300,400,528])
        ax.tick_params(labelsize=7,length=3,pad=2)
        ax.set_title(f'({chr(97+2*row+c)}) {"AE"[c]} channel',loc='left',fontsize=9,color=colors['truth'],pad=5)
        ax.text(0,1.19,f'Source {row+1} | gap {end-st:.0f} s | joint mismatch {mm[row]:.2e}',transform=ax.transAxes,fontsize=7,color='#4b596c',va='top')
        ax.set_ylabel('Normalized amplitude' if c==0 else '')
        if row==2:ax.set_xlabel('Time (s)')
        ax.grid(axis='y',alpha=.18,lw=.5)
        # A native-sample detail panel occupies the late, low-amplitude region.
        detail=ax.inset_axes([.55,.57,.425,.36]);lo=max(0,st-10);hi=min(527.5,end+10)
        detail.set_facecolor('#ffffff');detail.patch.set_alpha(.98)
        detail.axvspan(st,end,color=colors['shade'],lw=0)
        detail.plot(t,truth[row,c]/scale,color=colors['truth'],lw=1.05)
        detail.plot(t,gapped[row,c]/scale,color=colors['gap'],lw=.8,ls=(0,(4,2,1,2)))
        detail.plot(t,pred[row,c]/scale,color=colors['pred'],lw=.95,ls=(0,(3,2)))
        ix=(t>=lo)&(t<=hi);y=np.r_[truth[row,c,ix],pred[row,c,ix]]/scale
        margin=max(.035,float(np.ptp(y))*.12);detail.set(xlim=(lo,hi),ylim=(float(y.min()-margin),float(y.max()+margin)))
        detail.set_xticks([st,end]);detail.tick_params(labelsize=6,length=2,pad=1)
        detail.set_title('Gap detail',fontsize=6.5,pad=2,color='#4b596c')
        for spine in detail.spines.values():spine.set_visible(True);spine.set_color('#b6c2cf');spine.set_linewidth(.55)
        detail.grid(alpha=.13,lw=.4)
handles=[Line2D([],[],color=colors['noise'],lw=.85,label=r'Observation $d=Gh+n$'),Line2D([],[],color=colors['truth'],lw=1.5,label=r'Clean signal $h$'),Line2D([],[],color=colors['gap'],lw=1.1,ls=(0,(4,2,1,2)),label=r'Gapped signal $Gh$'),Line2D([],[],color=colors['pred'],lw=1.1,ls=(0,(3,2)),label=r'DGF reconstruction $\widehat h$')]
fig.legend(handles=handles,loc='outside upper center',ncols=2,frameon=False,columnspacing=2.2,handlelength=3,borderaxespad=.5)
for ext in ('pdf','svg','png'):fig.savefig(source/('latest_examples.'+ext),dpi=220,facecolor='white')
plt.close(fig)
# Preserve the full-test phase distributions; replace only the illustrative
# phase trajectory with the accurately reconstructed short-gap case above.
from scipy.signal import hilbert
metrics=np.load(HERE.parent/'manuscript_unified_20261015/publication_metrics.npz')
fig,ax=plt.subplots(1,2,figsize=(7.35,3.05),layout='constrained')
for c in range(2):
    values=metrics['phase_'+str('AE'[c])];ax[0].step(np.sort(values),np.arange(1,len(values)+1)/len(values),where='post',color=['#cc5f20','#14847b'][c],lw=1.4,label=f'{"AE"[c]} channel')
ax[0].set_xscale('symlog',linthresh=.2);ax[0].set(xlim=(0,180),ylim=(0,1.015),xlabel='Absolute phase error (deg)',ylabel='Fraction of test signals',title='(a) Phase error at signal-envelope peak');ax[0].legend(frameon=False);ax[0].grid(alpha=.15,lw=.6)
z=hilbert(truth[0,0]);w=hilbert(pred[0,0]);phase=np.unwrap(np.angle(z));difference=np.angle(w*z.conj());visible=abs(z)>.1*abs(z).max();st=mask[0].argmax()/2;end=st+mask[0].sum()/2
ax[1].axvspan(st,end,color=colors['shade'],lw=0);ax[1].plot(t,np.where(visible,phase,np.nan),color=colors['truth'],lw=1.35,label='Clean signal');ax[1].plot(t,np.where(visible,phase+difference,np.nan),color=colors['pred'],ls='--',lw=1.1,label='DGF');ax[1].axvline(abs(z).argmax()/2,color='#718096',ls=':',lw=.8)
ax[1].set(xlim=(max(0,st-50),min(528,end+80)),xlabel='Time (s)',ylabel='Analytic phase (rad)',title=f'(b) A channel | seed {seeds[chosen[0]]}');ax[1].legend(frameon=False);ax[1].grid(alpha=.15,lw=.6)
for ext in ('pdf','svg','png'):fig.savefig(source/('latest_phase.'+ext),dpi=220,facecolor='white')
plt.close(fig)
tex=(source/'DGF.tex').read_text()
# Locate the caption by the figure label, preserving every other source change.
label=tex.index(r'\label{fig:examples}');start=tex.rfind(r'\caption{',0,label)
assert start>0 and tex[start:label].count(r'\caption{')==1
caption=r'\caption{Selected successful reconstructions for 34-, 45- and 52-s gaps in the $A$ (left) and $E$ (right) channels. Each panel compares the noisy observation $d=Gh+n$, clean signal $h$, gapped clean signal $Gh$ and DGF reconstruction $\widehat h$ over the full 528-s analysis window; insets resolve the gap. Shading marks signal suppression, with noise retained. The examples have envelope peaks inside the gap and at least four dominant-mode damping times after the response peak within the window. All curves use one common clean-signal amplitude normalization per source and connect the original 0.5-s samples. Source seeds, from top to bottom, are 2038201123, 2038201889 and 2038204013. Full-population statistics use all 6,000 test sources.}'
tex=tex[:start]+caption+tex[label:]
onset="PyCBC's short pre-reference taper supplies a smooth numerical onset; the modeled damped-mode segment retains the stated excitation normalization."
assert "PyCBC's pre-reference exponential ring-up" in tex
begin=tex.index(r'Figure~\ref{fig:examples} illustrates');finish=tex.index('\n',begin)
paragraph=r'Figure~\ref{fig:examples} shows accurate reconstructions across 34-, 45- and 52-s gaps, with noise-weighted mismatches $6.97\times10^{-4}$, $4.47\times10^{-4}$ and $5.70\times10^{-4}$. The gaps contain 46.6\%, 38.0\% and 47.5\% of the respective joint signal squared norms. Their gap-local relative errors are 2.9\%, 5.7\% and 3.5\%, and both channels retain the decaying oscillations through the missing intervals. The windows extend for 4.3--7.3 dominant-mode damping times after the response peak. These selected successful cases complement the complete-population tables and distributions; central and upper-tail cases are shown in Appendix~\ref{app:complementary}.'
tex=tex[:begin]+paragraph+tex[finish:]
tex=tex.replace('source 2038201885, also shown in Fig.~\\ref{fig:examples}(c)','source 2038201123, also shown in Fig.~\\ref{fig:examples}(a)')
assert 'source 2038201123, also shown' in tex
tex=tex.replace(r'approximately 39\%--43\% of the signal squared norm',r'approximately 38\%--48\% of the signal squared norm')
anchor='The analysis uses event-centered 528-s windows and single 32--52-s gaps.'
assert anchor in tex
tex=tex.replace(anchor,anchor+' The specified pre-reference ring-up fixes the signal history entering the TDI delays; continuous merger histories provide a further robustness test.')
anchor=r'Figure~\ref{fig:complementary_examples} gives sources'
assert anchor in tex
selection_text=r'The successful examples in Fig.~\ref{fig:examples} are selected for clear oscillations and small amplitude and phase residuals. The eligible records contain at least four $\tau_{220}$ after the joint response peak, have $\rho\ge10$ and have envelope peaks inside the gap. Each displayed source has global and gap-local relative errors below 8\% and per-channel peak phase errors below $1.5^\circ$. Low-mismatch cases are chosen across the gap-duration ranges 32--38, 39--45 and 46--52~s; their seeds and detailed selection conditions are recorded with the figure data.'
tex=tex.replace(anchor,selection_text+'\n\n'+anchor)
(source/'DGF.tex').write_text(tex)
audit=dict(checkpoint_sha256=report['checkpoint_sha256'],source_run=str(RUN),n_samples=1056,sample_spacing_seconds=.5,record_seconds=528,time_alignment=False,phase_alignment=False,display_interpolation=False,smoothing=False,selection=selection,all_population_statistics_unchanged=True,selected_indices=chosen.tolist(),selected_seeds=seeds[chosen].tolist(),joint_mismatch=mm.tolist(),input_equation_max_error=float(abs(observed-(gapped+noise)).max()),gapped_signal_inside_max=float(abs(gapped*mask[:,None]).max()),source_data_sha={str(p.relative_to(RUN)):sha(p) for p in [RUN/'fresh_test/truth.npy',RUN/'fresh_test/noise.npy',RUN/'variable_optimized_prediction.npy']})
(DEST/'figure4_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
