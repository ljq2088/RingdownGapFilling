from pathlib import Path
import json,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import hilbert
P=Path(__file__).resolve().parent;O=P/'source'
R=Path('/home/ljq/code/RingdownGapFilling_PRD');F=R/'FillingGap/results_prd_tail_20261015';T=F/'fresh_test'
a=np.load(P/'publication_metrics.npz');psd=np.load(R/'FillingGap/results_prd_physical_transfer_20261009/calibration_psd.npz')['psd'];asd=np.sqrt(psd);asd[:,0]=asd[:,1];asd[:,-1]=asd[:,-2]
chosen=np.array([2428,4884]);t=np.arange(1056)*.5
h=np.load(T/'truth.npy',mmap_mode='r')[chosen].astype(float);n=np.load(T/'noise.npy',mmap_mode='r')[chosen].astype(float);p=np.fft.irfft(np.fft.rfft(np.load(F/'variable_optimized_prediction.npy',mmap_mode='r')[chosen].astype(float))*asd,n=1056)
m=np.array([(t>=a['start'][i])&(t<a['start'][i]+a['length'][i]) for i in chosen]);g=h*(1-m[:,None]);d=g+n
np.savez_compressed(P/'figure_native_samples.npz',time=t,index=chosen,seeds=a['seeds'][chosen],truth=h,noise=n,mask=m,gapped_truth=g,observation=d,reconstruction=p)
sel=[dict(index=int(i),**{k:a[k][i].tolist() for k in a.files}) for i in chosen]
(P/'selected_examples.json').write_text(json.dumps(dict(selected=sel,display_processing='native 0.5-s samples; common clean-signal amplitude scale; no display smoothing, resampling, time/phase alignment or noise rescaling',selection='Resolved decaying A/E oscillations and small full-record and gap-local errors; full 528-s observations plus labelled waveform-detail intervals; population statistics retain all 6000 sources'),indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,'legend.fontsize':7.5,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'path.simplify':False})
C=dict(noise='#aab3bf',h='#1e334f',g='#14847b',p='#cc5f20',shade='#e6edf5')
def save(fig,name):
 for ext in ('pdf','png','svg'):fig.savefig(O/(name+'.'+ext),dpi=200,facecolor='white')
 plt.close(fig)
def grid(ax):ax.grid(alpha=.15,lw=.5)
fig=plt.figure(figsize=(7.35,5.25),layout='constrained')
gs=fig.add_gridspec(4,2,height_ratios=[1,1.15,1,1.15],hspace=.12)
fig.set_constrained_layout_pads(w_pad=.025,h_pad=.025,hspace=.08,wspace=.04)
handles=[Line2D([],[],color=C['noise'],lw=.8,label=r'Observation $Gh+n$'),Line2D([],[],color=C['h'],lw=1.35,label=r'Clean signal $h$'),Line2D([],[],color=C['g'],lw=.95,ls=(0,(4,2,1,2)),label=r'Gapped signal $Gh$'),Line2D([],[],color=C['p'],lw=1.1,ls=(0,(3,2)),label=r'DGF reconstruction $\widehat h$')]
for row,i in enumerate(chosen):
 st=a['start'][i];end=st+a['length'][i];s=abs(h[row]).max()
 for c in range(2):
  upper=fig.add_subplot(gs[2*row,c]);lower=fig.add_subplot(gs[2*row+1,c])
  for ax in (upper,lower):ax.axvspan(st,end,color=C['shade'],lw=0,zorder=0);grid(ax);ax.tick_params(labelsize=7)
  upper.plot(t,d[row,c]/s,color=C['noise'],lw=.4,alpha=.8)
  for v,col,lw,ls in [(h,C['h'],1.25,'-'),(g,C['g'],.8,(0,(4,2,1,2))),(p,C['p'],1.0,(0,(3,2)))]:
   for ax in (upper,lower):ax.plot(t,v[row,c]/s,color=col,lw=lw,ls=ls)
  upper.set(xlim=(0,528),xticks=[0,100,200,300,400,528],ylim=(-3.15,3.15))
  upper.set_title(f'({chr(97+2*row+c)}) Source {row+1} | {"AE"[c]} channel | gap {end-st:.0f} s',loc='left',pad=5)
  upper.text(.98,.92,'Full record',ha='right',va='top',transform=upper.transAxes,fontsize=6.5,color='#536477',bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=1))
  window=[(65,350),(0,300)][row]
  lower.set(xlim=window,ylim=(-1.1,1.1),xlabel='Time (s)',yticks=[-1,0,1])
  lower.text(.98,.9,'Waveform detail',ha='right',va='top',transform=lower.transAxes,fontsize=6.5,color='#536477')
  mantissa,exponent=f'{a["mismatch"][i]:.2e}'.split('e')
  lower.text(.98,.06,rf'$\mathcal{{M}}_{{AE}}={mantissa}\times10^{{{int(exponent)}}}$',ha='right',va='bottom',transform=lower.transAxes,fontsize=7,color='#536477')
  if c==0:
   upper.set_ylabel(r'Amplitude / $H_{\rm TDI}$');lower.set_ylabel(r'Amplitude / $H_{\rm TDI}$')
fig.legend(handles=handles,loc='outside upper center',ncols=2,frameon=False,handlelength=3,columnspacing=2)
save(fig,'latest_examples')
fig,axes=plt.subplots(2,2,figsize=(7.35,3.7),layout='constrained')
for row,i in enumerate(chosen):
 for c in range(2):
  ax=axes[row,c];s=abs(h[row]).max();ax.axvspan(a['start'][i],a['start'][i]+a['length'][i],color=C['shade'],lw=0)
  ax.semilogy(t,abs(h[row,c])/s,color=C['h'],lw=1.1);ax.semilogy(t,abs(p[row,c])/s,color=C['p'],ls='--',lw=1)
  ax.set(xlim=(0,528),ylim=(1e-4,1.5),xticks=[0,100,200,300,400,528],title=f'({chr(97+2*row+c)}) Source {row+1} | {"AE"[c]} channel');grid(ax)
  if c==0:ax.set_ylabel(r'$|h_c|/H_{\rm TDI}$')
  if row==1:ax.set_xlabel('Time (s)')
fig.legend(handles=[handles[1],handles[3]],loc='outside upper center',ncols=2,frameon=False);save(fig,'tdi_log_amplitudes_20261015')
fig,axes=plt.subplots(1,2,figsize=(7.35,2.7),layout='constrained')
for j,values in enumerate([a['time_mm'],a['mismatch']]):
 ax=axes[j]
 for ix,label,col,ls in [(np.ones(6000,bool),'All 6,000 sources',C['h'],'-'),(a['snr']>=10,r'$\rho\geq10$ (5,810 sources)',C['p'],'--')]:
  v=values[ix];ax.step(np.sort(v),np.arange(1,len(v)+1)/len(v),where='post',color=col,lw=1.25,ls=ls,label=label)
 ax.set(xscale='log',xlim=(values.min()*.8,2),ylim=(0,1.015),xlabel='Time-domain mismatch' if j==0 else 'Noise-weighted mismatch',ylabel='Fraction of sources',title=f'({chr(97+j)}) Waveform agreement');grid(ax);ax.legend(frameon=False,loc='lower right',fontsize=7)
save(fig,'latest_mismatch')
fig,axes=plt.subplots(1,2,figsize=(7.35,2.7),layout='constrained');ax=axes[0]
ax.scatter(a['snr'],a['mismatch'],s=5,color=C['p'],alpha=.3,rasterized=True);ax.axvline(10,color=C['h'],ls=':',lw=1);ax.set(xscale='log',yscale='log',xlabel='Network SNR',ylabel='Noise-weighted mismatch',title='(a) SNR dependence');grid(ax)
ax=axes[1]
for ix,label,col in [(a['inside'],'Peak inside gap',C['p']),(~a['inside'],'Peak outside gap',C['g'])]:
 v=a['mismatch'][ix];ax.step(np.sort(v),np.arange(1,len(v)+1)/len(v),where='post',color=col,lw=1.3,label=f'{label} (n={len(v)})')
ax.set(xscale='log',ylim=(0,1.015),xlabel='Noise-weighted mismatch',ylabel='Fraction of subgroup',title='(b) Gap location');ax.legend(frameon=False,fontsize=7);grid(ax);save(fig,'latest_snr_gap')
fig,ax=plt.subplots(1,2,figsize=(7.35,2.6),layout='constrained')
for c in range(2):
 v=a['phase_peak'][:,c];ax[0].step(np.sort(v),np.arange(1,6001)/6000,where='post',color=[C['p'],C['g']][c],lw=1.3,label=f'{"AE"[c]} channel')
ax[0].set_xscale('symlog',linthresh=.2);ax[0].set(xlim=(0,180),ylim=(0,1.015),xlabel='Absolute phase error (deg)',ylabel='Fraction of test sources',title='(a) Phase error at envelope peak');ax[0].legend(frameon=False);grid(ax[0])
z=hilbert(h[0,0]);w=hilbert(p[0,0]);phi=np.unwrap(np.angle(z));visible=abs(z)>.1*abs(z).max();i=chosen[0];st=a['start'][i];end=st+a['length'][i]
ax[1].axvspan(st,end,color=C['shade'],lw=0);ax[1].plot(t,np.where(visible,phi,np.nan),color=C['h'],lw=1.3,label='Clean signal');ax[1].plot(t,np.where(visible,phi+np.angle(w*z.conj()),np.nan),color=C['p'],ls='--',lw=1.1,label='DGF');ax[1].axvline(abs(z).argmax()*.5,color='#718096',ls=':',lw=.8)
ax[1].set(xlim=(max(0,st-50),min(528,end+80)),xlabel='Time (s)',ylabel='Analytic phase (rad)',title='(b) Source 1 | A channel');ax[1].legend(frameon=False);grid(ax[1]);save(fig,'latest_phase')
v=np.load(P/'error_spectra.npz');f=v['frequency'];fig,axes=plt.subplots(2,2,figsize=(7.35,3.8),layout='constrained',gridspec_kw={'height_ratios':[2,1]},sharex='col')
for c in range(2):
 ax=axes[0,c]
 for k,label,col,ls in [('input_error',r'Input error $d-h$',C['h'],'-'),('noise','Instrument noise','#8192a3','--'),('residual',r'Reconstruction error $\widehat h-h$',C['p'],'-')]:
  ax.loglog(f[1:-1],v[k][c,1:-1],color=col,ls=ls,lw=1.15,label=label)
 ax.set(title=f'({chr(97+c)}) {"AE"[c]} channel');grid(ax)
 if c==0:ax.set_ylabel(r'Error PSD (Hz$^{-1}$)')
 ax=axes[1,c];ratio=v['residual'][c,1:-1]/v['input_error'][c,1:-1]
 ax.semilogx(f[1:-1],10*np.log10(ratio),color=C['p'],lw=1.1);ax.axhline(0,color='#8192a3',ls=':',lw=.8)
 ax.set(xlabel='Frequency (Hz)',ylim=(-100,4),yticks=[0,-40,-80]);grid(ax)
 if c==0:ax.set_ylabel('Error-power ratio (dB)')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside upper center',ncols=3,frameon=False,columnspacing=1.3,fontsize=7);save(fig,'latest_residual_psd')
