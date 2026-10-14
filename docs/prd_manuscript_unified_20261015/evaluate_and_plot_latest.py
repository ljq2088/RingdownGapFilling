"""All publication diagnostics use the frozen 6000-source independent test.

No training or model selection occurs here. Decode saved whitened predictions,
audit the direct joint A/E overlap, and reuse these predictions in every figure.
"""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert, periodogram

HERE=Path(__file__).resolve().parent
ROOT=Path('/home/ljq/code/RingdownGapFilling_PRD')
sys.path.insert(0,str(ROOT))
from prd_conditioned.settings import OUT as RUN, CAL
from prd_physical.common import Preprocess, metric, summary, sha
from prd_variable.data import variable_mask, fixed_mask

result=json.loads((RUN/'final_results.json').read_text())
assert result['passed'] and result['n']==6000 and result['all_rows_retained']
assert result['selected']['selection']['checkpoint_sha256']==sha(RUN/'finetune/best.pth')
fresh=RUN/'fresh_test'
h=np.load(fresh/'truth.npy').astype(np.float64)
noise=np.load(fresh/'noise.npy').astype(np.float64)
seeds=np.load(fresh/'seeds.npy')
original_mask=np.load(fresh/'mask.npy')
assert h.shape==noise.shape==(6000,2,1056)
assert np.array_equal(seeds,np.arange(2038200000,2038206000))
with np.load(RUN/'per_example_test.npz') as z: official={k:z[k] for k in z.files}
cal=np.load(CAL); pre=Preprocess(cal['psd'],'white')
mask=variable_mask(original_mask,seeds,410000000)
pred=pre.decode(np.load(RUN/'variable_optimized_prediction.npy',mmap_mode='r'))
reference=pre.decode(np.load(RUN/'variable_previous_prediction.npy',mmap_mode='r'))
mm=metric(h,pred,cal['psd']); time_mm=metric(h,pred)
assert np.max(np.abs(mm-official['variable__optimized']))<1e-12
assert np.isfinite(pred).all() and np.isfinite(h).all()

energy=np.sum(h*h,axis=(1,2)); residual_energy=np.sum((pred-h)**2,axis=(1,2))
assert np.all(energy>0)
relative_error=np.sqrt(residual_energy/energy)
norm_bias=np.sqrt(np.sum(pred*pred,axis=(1,2))/energy)-1
zh=hilbert(h,axis=-1); zp=hilbert(pred,axis=-1)
envelope=np.sqrt(np.sum(np.abs(zh)**2,axis=1))
joint_peaks=envelope.argmax(axis=-1)
peak_in_gap=mask[np.arange(6000),joint_peaks].astype(bool)
masked_energy=np.sum(h*h*mask[:,None],axis=(1,2))/energy
gap_relative=np.sqrt(np.sum((pred-h)**2*mask[:,None],axis=(1,2))/np.sum(h*h*mask[:,None],axis=(1,2)))
channel_peaks=np.abs(zh).argmax(axis=-1)
phase=np.angle(zp[np.arange(6000)[:,None],np.arange(2)[None,:],channel_peaks]*
               zh[np.arange(6000)[:,None],np.arange(2)[None,:],channel_peaks].conj())
phase_abs=np.abs(phase)*180/np.pi
snr=official['snr']; strong=snr>=10
eligible=np.flatnonzero(peak_in_gap)
assert len(eligible)>=3
chosen=eligible[np.argsort(mm[eligible],kind='stable')[:3]]
order=np.argsort(mm,kind='stable')
diagnostic=order[[2999,5399,5999]]
def group(ix):
    return dict(n=int(np.sum(ix)),psd=summary(mm[ix]),time=summary(time_mm[ix]),
                relative_error=summary(relative_error[ix]),
                gap_relative_error=summary(gap_relative[ix]) if np.all(np.isfinite(gap_relative[ix])) else None,
                phase_A=summary(phase_abs[ix,0]),phase_E=summary(phase_abs[ix,1]))

report=dict(test_sources=6000,checkpoint_sha256=sha(RUN/'finetune/best.pth'),
            source_run=str(RUN),seeds=[int(seeds[0]),int(seeds[-1])],
            time_phase_alignment=False,all_sources_retained=True,
            direct_psd_audit_max_error=float(np.max(np.abs(mm-official['variable__optimized']))),
            primary=dict(psd=summary(mm),time=summary(time_mm),relative_error=summary(relative_error),
                         norm_bias=summary(norm_bias),phase_A=summary(phase_abs[:,0]),phase_E=summary(phase_abs[:,1])),
            snr_ge10=group(strong),peak_in_gap=group(peak_in_gap),peak_outside_gap=group(~peak_in_gap),
            snr=dict(min=float(snr.min()),max=float(snr.max()),median=float(np.median(snr)),p05=float(np.quantile(snr,.05)),p95=float(np.quantile(snr,.95))),
            gap_samples=[int(x) for x in np.unique(mask.sum(-1))],
            fixed_duration_results={name:result['results'][name]['optimized'] for name in ('fixed66','fixed100')},
            same_test_reference=result['results'],
            selected_cases=[dict(index=int(i),seed=int(seeds[i]),mismatch=float(mm[i]),
                masked_energy_fraction=float(masked_energy[i]),gap_seconds=float(mask[i].sum()/2),
                gap_start_seconds=float(mask[i].argmax()/2),phase_A_deg=float(phase_abs[i,0]),phase_E_deg=float(phase_abs[i,1])) for i in chosen],
            diagnostic_cases=[dict(index=int(i),seed=int(seeds[i]),mismatch=float(mm[i]),masked_energy_fraction=float(masked_energy[i])) for i in diagnostic])
report['snr_bins']=[]
for lo,hi in zip([0,5,10,20,50,100],[5,10,20,50,100,np.inf]):
    ix=(snr>=lo)&(snr<hi)
    report['snr_bins'].append(dict(low=lo,high=None if np.isinf(hi) else hi,n=int(ix.sum()),mean=float(mm[ix].mean()) if ix.any() else None))
report['provenance']={str(f.relative_to(RUN)):sha(f) for f in [RUN/'finetune/best.pth',RUN/'selection_frozen.json',RUN/'per_example_test.npz',fresh/'truth.npy',fresh/'noise.npy',fresh/'mask.npy',fresh/'seeds.npy',RUN/'variable_optimized_prediction.npy',RUN/'variable_previous_prediction.npy']}
(HERE/'latest_test_summary.json').write_text(json.dumps(report,indent=2))
np.savez_compressed(HERE/'publication_metrics.npz',seeds=seeds,snr=snr,psd_mismatch=mm,time_mismatch=time_mm,
                    relative_error=relative_error,norm_bias=norm_bias,phase_A=phase_abs[:,0],phase_E=phase_abs[:,1],
                    peak_in_gap=peak_in_gap,masked_energy_fraction=masked_energy)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,
 'legend.fontsize':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
COL=['#c66a22','#29988b']; TARGET='#263348'
def save(fig,name):
    for ext in ('pdf','svg','png'):fig.savefig(HERE/'source_edited'/(name+'.'+ext),dpi=200,facecolor='white')
    plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(7.35,3.05),constrained_layout=True)
for j,values in enumerate([time_mm,mm]):
    ax=axes[j]; ax.step(np.sort(values),np.arange(1,6001)/6000,where='post',color=COL[0],lw=1.6)
    ax.set(xscale='log',xlim=(max(1e-6,.8*values.min()),2),ylim=(0,1.015),
           xlabel='Time-domain mismatch' if j==0 else 'Noise-weighted mismatch',ylabel='Fraction of test signals')
    ax.set_title(f'({chr(97+j)}) DGF\nMean mismatch: {values.mean():.5f}',loc='left',fontsize=9,color=TARGET)
    ax.grid(alpha=.15,lw=.6)
save(fig,'latest_mismatch')

fig,axes=plt.subplots(3,2,figsize=(7.35,5.9),constrained_layout=True)
t=np.arange(1056)/2
for row,i in enumerate(chosen):
    st=mask[i].argmax()/2; end=st+mask[i].sum()/2
    scale=np.max(np.abs(h[i]))
    for c in range(2):
        ax=axes[row,c]; ax.axvspan(st,end,color='#dfe7f0',alpha=.75,lw=0)
        ax.plot(t,h[i,c]/scale,color=TARGET,lw=1.35,label='Clean signal')
        ax.plot(t,pred[i,c]/scale,color=COL[c],ls='--',lw=1.1,label='DGF')
        ax.set_xlim(max(0,st-65),min(528,end+100)); ax.set_ylim(-1.12,1.12)
        ax.set_title(f'({chr(97+row)}) {"AE"[c]} channel | seed {seeds[i]}\nMismatch (A/E): {mm[i]:.3g}',loc='left',fontsize=8.5,color=TARGET)
        ax.set_ylabel(r'Amplitude / $\max_{c,t}|h_c|$'); ax.grid(axis='y',alpha=.15,lw=.6)
        if row==0:ax.legend(loc='upper right',frameon=True,edgecolor='#d6dfe8')
        if row==2:ax.set_xlabel('Time (s)')
save(fig,'latest_examples')

f,pn=periodogram(noise,fs=2,axis=-1); _,pe=periodogram(pred-h,fs=2,axis=-1)
fig,axes=plt.subplots(1,2,figsize=(7.35,3.0),constrained_layout=True)
for c,ax in enumerate(axes):
    ax.loglog(f[1:-1],pe[:,c,1:-1].mean(0),color=COL[c],lw=1.5,label='Reconstruction residual')
    ax.loglog(f[1:-1],pn[:,c,1:-1].mean(0),color=TARGET,ls='--',lw=1,label='Instrument noise')
    ax.set(xlabel='Frequency (Hz)',ylabel=r'PSD (Hz$^{-1}$)',title=f'({chr(97+c)}) {"AE"[c]} channel')
    ax.grid(alpha=.15,lw=.6); ax.legend(loc='best',frameon=False,fontsize=7.5)
save(fig,'latest_residual_psd')

i=int(chosen[1]); c=0; z=zh[i,c]; peak=channel_peaks[i,c]
fig,axes=plt.subplots(1,2,figsize=(7.35,3.05),constrained_layout=True)
for c in range(2):
    axes[0].step(np.sort(phase_abs[:,c]),np.arange(1,6001)/6000,where='post',color=COL[c],lw=1.4,label=f'{"AE"[c]} channel')
axes[0].set_xscale('symlog',linthresh=.2)
axes[0].set(xlim=(0,180),ylim=(0,1.015),xlabel='Absolute phase error (deg)',ylabel='Fraction of test signals',title='(a) Phase error at signal-envelope peak')
axes[0].legend(frameon=False);axes[0].grid(alpha=.15,lw=.6)
c=0; z=zh[i,c]; target_phase=np.unwrap(np.angle(z)); delta=np.angle(zp[i,c]*z.conj())
visible=np.abs(z)>.1*np.abs(z).max();st=mask[i].argmax()/2;end=st+mask[i].sum()/2
ax=axes[1];ax.axvspan(st,end,color='#dfe7f0',alpha=.75,lw=0)
ax.plot(t,np.where(visible,target_phase,np.nan),color=TARGET,lw=1.35,label='Clean signal')
ax.plot(t,np.where(visible,target_phase+delta,np.nan),color=COL[0],ls='--',lw=1.1,label='DGF')
ax.axvline(channel_peaks[i,c]/2,color='#718096',ls=':',lw=.8)
ax.set(xlim=(max(0,st-50),min(528,end+80)),xlabel='Time (s)',ylabel='Analytic phase (rad)',title=f'(b) A channel | seed {seeds[i]}')
ax.grid(alpha=.15,lw=.6);ax.legend(frameon=False)
save(fig,'latest_phase')

fig,axes=plt.subplots(1,2,figsize=(7.35,3.1),constrained_layout=True)
ax=axes[0];ax.scatter(snr,mm,s=6,color=COL[0],alpha=.25,rasterized=True)
ax.set(xscale='log',yscale='log',xlabel='Network SNR',ylabel='Noise-weighted mismatch',title='(a) All independent test signals');ax.axvline(10,color=TARGET,ls=':',lw=1);ax.grid(alpha=.15,lw=.6)
ax=axes[1]
for ix,label,color in [(peak_in_gap,'Peak inside gap',COL[0]),(~peak_in_gap,'Peak outside gap',COL[1])]:
    v=mm[ix];ax.step(np.sort(v),np.arange(1,len(v)+1)/len(v),where='post',label=f'{label} (n={len(v)})',color=color,lw=1.4)
ax.set(xscale='log',ylim=(0,1.015),xlabel='Noise-weighted mismatch',ylabel='Fraction of subgroup',title='(b) Dependence on gap location');ax.legend(frameon=False,fontsize=7.5);ax.grid(alpha=.15,lw=.6)
save(fig,'latest_snr_gap')

fig,axes=plt.subplots(3,2,figsize=(7.35,5.9),constrained_layout=True)
for row,i in enumerate(diagnostic):
    st=mask[i].argmax()/2;end=st+mask[i].sum()/2;scale=np.max(np.abs(h[i]))
    for c in range(2):
        ax=axes[row,c];ax.axvspan(st,end,color='#dfe7f0',alpha=.75,lw=0)
        ax.plot(t,h[i,c]/scale,color=TARGET,lw=1.3,label='Clean signal');ax.plot(t,pred[i,c]/scale,color=COL[c],ls='--',lw=1.05,label='DGF')
        ax.set_xlim(max(0,st-65),min(528,end+100));ax.set_ylabel('Normalized amplitude')
        ax.set_title(f'Rank {[3000,5400,6000][row]}/6000 | {"AE"[c]} channel\nMismatch (A/E): {mm[i]:.4g}',loc='left',fontsize=8.5)
        ax.grid(axis='y',alpha=.15,lw=.6)
        if row==0:ax.legend(frameon=False,fontsize=7.5)
        if row==2:ax.set_xlabel('Time (s)')
save(fig,'latest_diagnostic_examples')

hist=json.loads((RUN/'finetune/history.json').read_text());hh=json.loads((RUN/'head_full/history.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(7.35,2.8),constrained_layout=True)
for ax,records,label,best in zip(axes,[hh,hist],['(a) Coefficient-head training','(b) Joint fine-tuning'],[18,6]):
    ax.plot([r['epoch'] for r in records],[r['validation']['score'] for r in records],color=COL[0],lw=1.2)
    ax.axvline(best,color=TARGET,ls=':',lw=.9,label='Selected epoch')
    ax.set(xlabel='Epoch',ylabel='Validation mean mismatch',title=label);ax.legend(frameon=False,fontsize=7.5);ax.grid(alpha=.15,lw=.6)
save(fig,'latest_training')
print(json.dumps({k:report[k] for k in ['primary','snr_ge10','peak_in_gap','snr','selected_cases']},indent=2))
