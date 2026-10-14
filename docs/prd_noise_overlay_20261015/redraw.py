from pathlib import Path
import json,re,zipfile,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator,NullFormatter
from scipy.signal import hilbert
HERE=Path(__file__).resolve().parent;source=HERE/'source'
audit=json.loads((HERE/'figure_audit.json').read_text());sel=audit['selected']
n=np.load(HERE/'figure4_native_samples.npz');t=n['time'];truth=n['truth'];pred=n['reconstruction'];mask=n['mask'];observed=n['observation'];gapped=n['gapped_truth'];mm=n['mismatch'];seeds=n['seeds']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'legend.fontsize':7.5,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'path.simplify':False})
colors=dict(noise='#aab3bf',truth='#1e334f',gap='#14847b',pred='#cc5f20',shade='#e6edf5')
def save(fig,name):
 for ext in ['pdf','svg','png']:fig.savefig(source/(name+'.'+ext),dpi=220,facecolor='white')
 plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(7.35,5.65),layout='constrained')
fig.set_constrained_layout_pads(w_pad=.035,h_pad=.04,hspace=.10,wspace=.05)
for row in range(2):
 st=mask[row].argmax()/2;end=st+mask[row].sum()/2;scale=abs(truth[row]).max()
 for c in range(2):
  ax=axes[row,c];ax.axvspan(st,end,color=colors['shade'],lw=0,zorder=0)
  ax.plot(t,observed[row,c]/scale,color=colors['noise'],lw=.5,alpha=.60,zorder=1)
  ax.plot(t,truth[row,c]/scale,color=colors['truth'],lw=1.5,zorder=3)
  ax.plot(t,gapped[row,c]/scale,color=colors['gap'],lw=1,ls=(0,(4,2,1,2)),zorder=4)
  ax.plot(t,pred[row,c]/scale,color=colors['pred'],lw=1.1,ls=(0,(3,2)),zorder=5)
  extent=max(1.18,float(abs(observed[row,c]/scale).max())*1.04)
  ax.set(xlim=(0,528),ylim=(-extent,extent),xticks=[0,100,200,300,400,528],ylabel='Normalized amplitude' if c==0 else '',xlabel='Time (s)' if row==1 else '')
  ax.set_title(f'({chr(97+row*2+c)}) {"AE"[c]} channel | gap {end-st:.0f} s\nJoint mismatch {mm[row]:.2e}',loc='left',fontsize=8.5,color=colors['truth'],pad=6)
  ax.tick_params(labelsize=7,length=3,pad=2);ax.grid(axis='y',alpha=.18,lw=.5)
  detail=ax.inset_axes([.55,.60,.425,.34]);lo=max(0,st-10);hi=min(527.5,end+10)
  detail.set_facecolor('white');detail.axvspan(st,end,color=colors['shade'],lw=0)
  for values,color,ls,lw in [(truth,colors['truth'],'-',1.1),(gapped,colors['gap'],(0,(4,2,1,2)),.8),(pred,colors['pred'],(0,(3,2)),1.)]:detail.plot(t,values[row,c]/scale,color=color,lw=lw,ls=ls)
  ix=(t>=lo)&(t<=hi);y=np.r_[truth[row,c,ix],pred[row,c,ix]]/scale;margin=max(.035,float(np.ptp(y))*.12)
  detail.set(xlim=(lo,hi),ylim=(float(y.min()-margin),float(y.max()+margin)));detail.set_xticks([st,end]);detail.tick_params(labelsize=6,length=2,pad=1)
  detail.set_title('Gap detail',fontsize=6.5,pad=2,color='#4b596c')
  for spine in detail.spines.values():spine.set_visible(True);spine.set_color('#b6c2cf');spine.set_linewidth(.55)
  detail.grid(alpha=.13,lw=.4)
handles=[Line2D([],[],color=colors['noise'],lw=.85,label=r'Observation $d=Gh+n$'),Line2D([],[],color=colors['truth'],lw=1.5,label=r'Clean signal $h$'),Line2D([],[],color=colors['gap'],lw=1.1,ls=(0,(4,2,1,2)),label=r'Gapped signal $Gh$'),Line2D([],[],color=colors['pred'],lw=1.1,ls=(0,(3,2)),label=r'DGF reconstruction $\widehat h$')]
fig.legend(handles=handles,loc='outside upper center',ncols=2,frameon=False,columnspacing=2,handlelength=3)
save(fig,'latest_examples')
# Keep the population CDF exactly as previously published in this revision.
metrics=np.load(HERE.parent/'manuscript_unified_20261015/publication_metrics.npz')
fig,ax=plt.subplots(1,2,figsize=(7.35,3.05),layout='constrained')
for c in range(2):
 v=metrics['phase_'+str('AE'[c])];ax[0].step(np.sort(v),np.arange(1,len(v)+1)/len(v),where='post',color=['#cc5f20','#14847b'][c],lw=1.4,label=f'{"AE"[c]} channel')
ax[0].set_xscale('symlog',linthresh=.2);ax[0].set(xlim=(0,180),ylim=(0,1.015),xlabel='Absolute phase error (deg)',ylabel='Fraction of test signals',title='(a) Phase error at signal-envelope peak');ax[0].legend(frameon=False);ax[0].grid(alpha=.15,lw=.6)
z=hilbert(truth[0,0]);w=hilbert(pred[0,0]);phase=np.unwrap(np.angle(z));difference=np.angle(w*z.conj());visible=abs(z)>.1*abs(z).max();st=mask[0].argmax()/2;end=st+mask[0].sum()/2
ax[1].axvspan(st,end,color=colors['shade'],lw=0);ax[1].plot(t,np.where(visible,phase,np.nan),color=colors['truth'],lw=1.35,label='Clean signal');ax[1].plot(t,np.where(visible,phase+difference,np.nan),color=colors['pred'],ls='--',lw=1.1,label='DGF');ax[1].axvline(abs(z).argmax()/2,color='#718096',ls=':',lw=.8)
ax[1].set(xlim=(max(0,st-50),min(528,end+80)),xlabel='Time (s)',ylabel='Analytic phase (rad)',title=f'(b) A channel | seed {seeds[0]}');ax[1].legend(frameon=False);ax[1].grid(alpha=.15,lw=.6)
save(fig,'latest_phase')
fig,axes=plt.subplots(2,2,figsize=(7.35,4.8),layout='constrained');fig.set_constrained_layout_pads(w_pad=.035,h_pad=.04,hspace=.075,wspace=.04)
for row in range(2):
 scale=abs(truth[row]).max();st=mask[row].argmax()/2;end=st+mask[row].sum()/2
 for c,ax in enumerate(axes[row]):
  ax.axvspan(st,end,color=colors['shade'],lw=0)
  ax.semilogy(t,np.where(abs(truth[row,c])>0,abs(truth[row,c])/scale,np.nan),color=colors['truth'],lw=1.15)
  ax.semilogy(t,np.where(abs(pred[row,c])>0,abs(pred[row,c])/scale,np.nan),color=colors['pred'],lw=1.05,ls=(0,(4,2)))
  ax.set(xlim=(0,528),ylim=(1e-5,1.5),xticks=[0,100,200,300,400,528],ylabel=r'$|h_c|/H_{\rm TDI}$' if c==0 else '',xlabel='Time (s)' if row==1 else '')
  ax.set_title(f'({chr(97+2*row+c)}) Source {row+1} | {"AE"[c]} channel',loc='left',pad=6);ax.grid(alpha=.15,lw=.5)
  ax.yaxis.set_major_locator(LogLocator(base=10,numticks=6));ax.yaxis.set_minor_formatter(NullFormatter())
fig.legend(handles=[Line2D([],[],color=colors['truth'],lw=1.3,label=r'Clean signal $|h_c|$'),Line2D([],[],color=colors['pred'],ls=(0,(4,2)),lw=1.1,label=r'DGF reconstruction $|\widehat h_c|$'),Patch(facecolor=colors['shade'],label='Gap')],loc='outside upper center',ncols=3,frameon=False,handlelength=2.5,columnspacing=1.5)
save(fig,'tdi_log_amplitudes_20261015')
