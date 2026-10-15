from pathlib import Path
import json, numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
reports={r['index']:r for r in json.loads((P/'frequency_screen.json').read_text())}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'path.simplify':False})
fig,axes=plt.subplots(4,2,figsize=(8.6,7.2),layout='constrained',gridspec_kw={'height_ratios':[.6,.6,1.35,1.35]})
selected=[]
for j,idx in enumerate([3322,5712]):
 d=np.load(P/f'candidate_{idx}.npz');r=reports[idx].copy();t=d['time'];use=(t>=0)&(t<=240);source=d['source'][:,use];rec=d['reconstruction'][:,use];noise=d['projected_noise'][:,use]
 r['source_display_seconds']=[0,240];r['display_joint_relative_error']=float(np.linalg.norm(rec-source)/np.linalg.norm(source));selected.append(r)
 tt=np.arange(1056)*.5;mask=d['mask'];start=tt[mask][0];stop=tt[mask][-1]+.5
 for c in range(2):
  ax=axes[c,j];ax.axvspan(start,stop,color='#e3ebf4',lw=0);ax.plot(tt,d['observation'][c]/1e-18,color='#9eabb8',lw=.35,alpha=.8);ax.set(xlim=(0,528),xticks=[0,132,264,396,528],ylabel='TDI '+'AE'[c]+r' ($10^{-18}$)');ax.tick_params(labelsize=7);ax.grid(alpha=.12);ax.text(.98,.8,r'$d=Gh+n$',transform=ax.transAxes,ha='right',fontsize=7,color='#586977',bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=1))
  if c==0:ax.set_title(('(a) Lower frequency' if j==0 else '(b) Higher frequency')+f' | '+rf'$f_{{220}}={r["frequency_hz"]*1000:.1f}$ mHz',fontsize=10)
  if c==1:ax.set_xlabel('Record time (s)',fontsize=8)
 exponent=int(np.floor(np.log10(abs(source).max())));scale=10.**exponent;r['strain_axis_exponent']=exponent
 for c in range(2):
  ax=axes[c+2,j];clean=source[c]/scale;noisy=(source[c]+noise[c])/scale
  ax.plot(t[use],noisy,color='#9eabb8',lw=2.5,alpha=.85,label=r'Strain + projected noise $s+Pn$');ax.fill_between(t[use],clean,noisy,color='#9eabb8',alpha=.3,lw=0)
  ax.plot(t[use],clean,color='#203d59',lw=1.4,label=r'Injected strain $s$');ax.plot(t[use],rec[c]/scale,color='#d16b2e',ls=(0,(4,2)),lw=1.3,label=r'Conditional reconstruction $P\widehat h$');ax.axhline(0,color='#b7bfc7',lw=.5,zorder=0);ax.set(xlim=(0,240),xticks=[0,60,120,180,240],xlabel='Source time (s)',ylabel=('$h_+$' if c==0 else r'$h_\times$')+rf' ($10^{{{exponent}}}$)');ax.grid(alpha=.13)
  if c==0:ax.text(.97,.9,rf'$\tau_{{220}}={r["tau_seconds"]:.1f}$ s',transform=ax.transAxes,ha='right',fontsize=8,color='#586977')
fig.legend(*axes[2,0].get_legend_handles_labels(),loc='outside lower center',ncols=3,frameon=False,fontsize=8)
for ext in ['pdf','png','svg']:fig.savefig(P/f'conditional_frequency_contrast.{ext}',dpi=220)
(P/'frequency_contrast_report.json').write_text(json.dumps(dict(sources=selected,selection='Two illustrative good-recovery cases at separated dominant-mode frequencies. Screened 8 low-frequency and 8 high-frequency cases with rho>=10, direct mismatch<0.005 and gap-relative error<0.15. Conditional source errors also assessed; no population metrics changed.',display='Original native 0.5-s samples, common source-time axis 0..240 observer-frame seconds from amplitude reference epoch, original noise amplitudes.'),indent=2))
print(json.dumps(selected,indent=2))
