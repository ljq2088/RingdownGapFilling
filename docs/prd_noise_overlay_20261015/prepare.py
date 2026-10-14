from pathlib import Path
import zipfile,shutil
here=Path(__file__).resolve().parent
backup=Path.home()/'Downloads/RingdownGapFilling___PRD_revision_2026_10_09 (22).zip'
shutil.copy2(backup,here/'overleaf_before.zip')
with zipfile.ZipFile(backup) as z:z.extractall(here/'source')
assert (here/'source/DGF.tex').read_bytes()==(here.parent/'figure_cleanup_20261015/source/DGF.tex').read_bytes()
old=(here.parent/'figure_cleanup_20261015/redraw.py').read_text()
header=old[:old.index('fig=plt.figure')]
phase=old[old.index('# Keep the population CDF'):old.index('# These rank-based diagnostics')]
log=old[old.index('fig,axes=plt.subplots(3,2,figsize=(7.35,6.8)'):old.index('texpath=source/')]
log=log.replace('plt.subplots(3,2,figsize=(7.35,6.8)','plt.subplots(2,2,figsize=(7.35,4.8)').replace('range(3)','range(2)').replace('row==2','row==1')
main=r'''
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
'''
(here/'redraw.py').write_text(header+main+phase+log)
compile_script=(here.parent/'figure_cleanup_20261015/stage_compile.py').read_text().replace('ringdown_figure_cleanup_20261015','ringdown_noise_overlay_20261015')
(here/'stage_compile.py').write_text(compile_script)
