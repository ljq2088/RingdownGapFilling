from pathlib import Path
import zipfile,json,hashlib,shutil
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;B=P.parent/'ringdown_source_noise_20261015'
with zipfile.ZipFile(Path.home()/'Downloads/RingdownGapFilling___PRD_revision_2026_10_09 (54).zip') as z:z.extractall(P/'source')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'path.simplify':False})
a=np.load(B/'conditional_source_arrays.npz');native=np.load(P.parent/'ringdown_frequency_examples_20261015/figure_native_samples.npz');r=json.loads((B/'conditional_source_report.json').read_text())['sources']
cases=[]
for j in range(3):
 key=f'source{j+1}_';cases.append(dict(time=a[key+'time']/r[j]['tau_seconds'],source=a[key+'target'],rec=a[key+'conditional_reconstruction'],pn=a[key+'projected_noise'],truth=native['truth'][j],observation=native['observation'][j],ae_rec=native['reconstruction'][j],mask=native['mask'][j],freq=r[j]['frequency_hz'],tau=r[j]['tau_seconds']))
contrast=[];rr=json.loads((B/'frequency_contrast_report.json').read_text())['sources']
for idx,r in zip([3322,5712],rr):
 d=np.load(B/f'candidate_{idx}.npz');u=(d['time']>=0)&(d['time']<=240);contrast.append(dict(time=d['time'][u],source=d['source'][:,u],rec=d['reconstruction'][:,u],pn=d['projected_noise'][:,u],truth=d['truth'],observation=d['observation'],ae_rec=d['ae_reconstruction'],mask=d['mask'],freq=r['frequency_hz'],tau=r['tau_seconds']))
def draw(cases,name,normalized):
 cols=len(cases);fig,axes=plt.subplots(4,cols,figsize=(9.4 if cols==3 else 8.6,7.5),layout='constrained',gridspec_kw={'height_ratios':[.95,.95,1.25,1.25]})
 for j,d in enumerate(cases):
  tt=np.arange(1056)*.5;mask=d['mask'];start=tt[mask][0];stop=tt[mask][-1]+.5
  for c in range(2):
   ax=axes[c,j];ax.axvspan(start,stop,color='#e3ebf4',lw=0,zorder=0)
   ax.plot(tt,d['observation'][c]/1e-18,color='#8b99a7',lw=.38,alpha=.5,zorder=1,label=r'Observation $Gh+n$')
   ax.plot(tt,d['truth'][c]/1e-18,color='#203d59',lw=1.1,zorder=3,label=r'Clean signal $h$')
   ax.plot(tt,(d['truth'][c]*(1-mask))/1e-18,color='#287d72',lw=.85,ls='-.',zorder=2,label=r'Gapped signal $Gh$')
   ax.plot(tt,d['ae_rec'][c]/1e-18,color='#d16b2e',lw=1.1,ls=(0,(4,2)),zorder=4,label=r'DGF reconstruction $\widehat h$')
   ax.set(xlim=(0,528),xticks=[0,264,528],ylabel='TDI '+'AE'[c]+r' ($10^{-18}$)');ax.tick_params(labelsize=7);ax.grid(alpha=.12)
   if c==0:ax.set_title((f'Source {j+1}' if normalized else ['Lower frequency','Higher frequency'][j])+rf' | $f_{{220}}={1000*d["freq"]:.1f}$ mHz',fontsize=9)
   if c==1:ax.set_xlabel('Record time (s)',fontsize=8)
  exponent=int(np.floor(np.log10(abs(d['source']).max())));scale=10.**exponent
  for c in range(2):
   ax=axes[c+2,j];t=d['time'];clean=d['source'][c]/scale;noisy=(d['source'][c]+d['pn'][c])/scale
   ax.plot(t,noisy,color='#9eabb8',lw=2.4,alpha=.8,label=r'Strain + projected noise $s+Pn$');ax.fill_between(t,clean,noisy,color='#9eabb8',alpha=.3,lw=0)
   ax.plot(t,clean,color='#203d59',lw=1.35,label=r'Injected strain $s$');ax.plot(t,d['rec'][c]/scale,color='#d16b2e',ls=(0,(4,2)),lw=1.25,label=r'Conditional reconstruction $P\widehat h$')
   ax.axhline(0,color='#b7bfc7',lw=.5,zorder=0);ax.set(xlim=(0,4 if normalized else 240),xlabel=r'Time $t/\tau_{220}$' if normalized else 'Source time (s)',ylabel=('$h_+$' if c==0 else r'$h_\times$')+rf' ($10^{{{exponent}}}$)');ax.grid(alpha=.13)
   if not normalized:ax.set_xticks([0,60,120,180,240])
   if c==0 and not normalized:ax.text(.97,.9,rf'$\tau_{{220}}={d["tau"]:.1f}$ s',transform=ax.transAxes,ha='right',fontsize=8,color='#586977')
 fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside upper center',ncols=4,frameon=False,fontsize=8)
 fig.legend(*axes[2,0].get_legend_handles_labels(),loc='outside lower center',ncols=3,frameon=False,fontsize=8)
 for ext in ['pdf','png']:fig.savefig(P/f'{name}.{ext}',dpi=210)
 shutil.copyfile(P/f'{name}.pdf',P/'source'/f'{name}.pdf');plt.close(fig)
draw(cases,'conditional_source_strain',True);draw(contrast,'conditional_frequency_contrast',False)
f=P/'source/DGF.tex';s=f.read_text(encoding='utf-8')
s=s.replace('The upper two rows show the full $A/E$ observations $d=Gh+n$ in fractional-frequency units of $10^{-18}$; blue shading marks the signal-suppression interval.','The upper two rows overlay the full $A/E$ observations $d=Gh+n$ (gray), clean signal $h$ (blue), gapped signal $Gh$ (green dash-dotted) and direct DGF reconstruction $\\widehat h$ (orange dashed) on the same axes, in fractional-frequency units of $10^{-18}$; blue shading marks the signal-suppression interval.')
s=s.replace('The upper rows show the complete 528-s $A/E$ observations in fractional-frequency units of $10^{-18}$, with the signal-suppression intervals shaded.','The upper rows overlay the complete 528-s $A/E$ observations, clean signals, gapped signals and direct DGF reconstructions, with the colors and line styles of Fig.~\\ref{fig:conditional_strain}. Their ordinates give fractional frequency in units of $10^{-18}$, and shading marks the signal-suppression intervals.')
assert 'upper two rows overlay' in s and 'upper rows overlay' in s
f.write_text(s,encoding='utf-8',newline='\n')
f=P/'source/reproducibility_manifest.json';v=json.loads(f.read_text(encoding='utf-8'))
for name in ['conditional_source_strain.pdf','conditional_frequency_contrast.pdf']:v['figure_sha256'][name]=hashlib.sha256((P/'source'/name).read_bytes()).hexdigest()
v['conditional_source_reconstruction']['noise_display']+=' Updated upper A/E panels overlay observation, truth, gapped truth and frozen direct DGF output on the same axes; lower panels overlay strain with projected noise and conditional reconstruction. No sample values, noise amplitudes or evaluation statistics changed.'
v['conditional_frequency_contrast']['display']+=' Upper A/E panels overlay noisy observation, truth, gapped truth and direct reconstruction.'
f.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8',newline='\n')
