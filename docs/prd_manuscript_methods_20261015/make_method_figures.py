from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
import os
OUT=Path(__file__).resolve().parent/'source_edited'
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'dejavusans'})
C={'ink':'#233348','muted':'#657389','blue':'#2c65a2','teal':'#237f83','violet':'#7661a3','orange':'#b97027','line':'#53647b','light':'#f5f7fa'}
FILL={'blue':'#edf3fa','teal':'#edf7f5','violet':'#f3eff9','orange':'#fff4e7','gray':'#f3f5f8'}
def setup(H):
 fig=plt.figure(figsize=(7.35,7.35*H/15));ax=fig.add_axes([.005,.005,.99,.99]);ax.set(xlim=(0,15),ylim=(0,H));ax.axis('off');return fig,ax

def panel(ax,x,y,w,h,title):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.01,rounding_size=.13',edgecolor='#dce3ec',facecolor='#fbfcfe',lw=.8,zorder=0))
 ax.text(x+.18,y+h-.30,title,ha='left',va='center',fontsize=9.2,fontweight='bold',color=C['ink'])
def card(ax,x,y,w,h,title,lines=(),color='blue',fs=8.1):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.01,rounding_size=.12',edgecolor=C.get(color,'#94a2b6'),facecolor=FILL[color],lw=.95,zorder=2))
 title_artist=ax.text(x+w/2,y+h-.26,title,ha='center',va='center',fontsize=8.65,fontweight='bold',color=C.get(color,C['ink']),zorder=3);title_artist._card_width=w-.22
 if lines:
  top=y+h-.57; bottom=y+.19
  ys=np.linspace(top,bottom,len(lines)) if len(lines)>1 else [(top+bottom)/2]
  for t,yy in zip(lines,ys):
   artist=ax.text(x+w/2,yy,t,ha='center',va='center',fontsize=fs,color=C['ink'],zorder=3);artist._card_width=w-.22
def arrow(ax,points,color='line',dashed=False):
 color=C[color];ls=(0,(3,2)) if dashed else '-'
 for a,b in zip(points[:-2],points[1:-1]):ax.plot([a[0],b[0]],[a[1],b[1]],lw=1.15,color=color,ls=ls,zorder=1)
 ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=8,lw=1.15,color=color,linestyle=ls,zorder=1,shrinkA=0,shrinkB=1))
def save(fig,name):
 if os.environ.get('FIGURE_ONLY') and not name.startswith(os.environ['FIGURE_ONLY']):plt.close(fig);return
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 for artist in fig.axes[0].texts:
  if hasattr(artist,'_card_width'):
   width=artist.get_window_extent(renderer).width;limit=artist.axes.transData.transform((artist._card_width,0))[0]-artist.axes.transData.transform((0,0))[0]
   if width>limit:
    fs=artist.get_fontsize()*limit/width*.985;artist.set_fontsize(fs)
    if fs<6.65:print('SMALL_TEXT',name,round(fs,2),artist.get_text())
 fig.savefig(OUT/(name+'.pdf'),facecolor='white');fig.savefig(OUT/(name+'.svg'),facecolor='white');fig.savefig(OUT/(name+'.png'),dpi=180,facecolor='white');plt.close(fig)

fig,ax=setup(10)
panel(ax,.07,5.58,14.86,4.35,'a  |  DGF encoder-decoder: temporal context + oscillatory features')
card(ax,.35,7.47,2.25,1.76,'A/E observation',[],'gray',7.65)
ax.text(1.475,8.60,r'$d=Gh+n$',ha='center',fontsize=8.2,color=C['ink'])
ax.text(1.475,8.30,'2 × 1056 samples',ha='center',fontsize=7.6,color=C['ink'])
ax.text(1.475,8.06,'528 s at 2 Hz',ha='center',fontsize=7.6,color=C['ink'])
# Schematic noisy waveform icon within the observation card.
t=np.linspace(0,1,180); clean=np.sin(37*t)*np.exp(-3*t);rng=np.random.default_rng(18);noise=.12*rng.normal(size=len(t));masked=(t>.28)&(t<.39);wave=clean.copy();wave[masked]=0;wave+=noise
ax.axvspan(.35+.18+.28*1.88,.35+.18+.39*1.88,ymin=7.53/10,ymax=7.92/10,facecolor='#efb4ae',alpha=.5,zorder=3)
ax.plot(.53+t*1.88,7.72+.13*wave,color=C['ink'],lw=.6,zorder=4)
# Move waveform below the text baseline.
card(ax,.35,6.05,2.25,.99,'PSD whitening',[r'$u=Wd$','Noise-only PSD'],'gray',7.65)
arrow(ax,[(1.47,7.47),(1.47,7.04)])
card(ax,3.22,8.02,3.12,1.17,'Time-frequency branch',['RMS-scaled Morlet CWT',r'$|C|,;\arg C$: 2 × 16 × 264'],'blue',8)
card(ax,6.69,8.02,3.25,1.17,'Mixing + adapter',['Temporal / scale mixing','Conv. + physical-band CWT',r'8 × 32 × 64'],'blue',7.75)
card(ax,3.22,6.11,6.72,1.32,'Multiresolution temporal branch',['Observed mean / std normalization → db4 subbands','8 subbands × 1056 samples → overlapping patches','32 patches × 64 samples; stride 32 → 8 × 32 × 64'],'violet',7.95)
arrow(ax,[(2.6,6.55),(2.88,6.55),(2.88,8.60),(3.22,8.60)],'blue')
arrow(ax,[(2.88,6.55),(3.22,6.55)],'violet')
arrow(ax,[(6.34,8.60),(6.69,8.60)],'blue')
ax.add_patch(Circle((10.47,7.72),.20,facecolor='white',edgecolor=C['line'],lw=1,zorder=2));ax.text(10.47,7.72,'+',ha='center',va='center',fontsize=13,color=C['ink'],zorder=3)
arrow(ax,[(9.94,8.60),(10.47,8.60),(10.47,7.92)],'blue')
arrow(ax,[(9.94,6.77),(10.47,6.77),(10.47,7.52)],'violet')
card(ax,11.02,7.09,3.57,2.10,'Fusion + encoder',['Add branch feature arrays','Token / conv. / position encoding','Residual Conv2d','16 channel-wise attention blocks'],'teal',7.65)
arrow(ax,[(10.67,7.72),(11.02,7.72)])
card(ax,11.02,5.85,3.57,1.00,'Learned waveform decoder',['Patch map → overlap average','Sum bands → Conv1d → rescale'],'teal',7.45)
arrow(ax,[(12.81,7.09),(12.81,6.85)])
ax.text(4.65,5.88,'Array sizes are per A/E channel; batch dimension omitted.',fontsize=7.25,color=C['muted'],ha='center')

panel(ax,.07,.10,14.86,5.02,'b  |  Joint A/E refinement and noise-conditioned reconstruction')
card(ax,.35,2.79,3.02,1.51,'Mask + A/E refiners',['Mask residual: width 24','Joint A/E residual: width 32','Dilations: 1, 2, 4, 8, 16, 32'],'orange',7.7)
# Explicit continuation from the upper decoder, routed outside cards.
arrow(ax,[(12.81,5.85),(12.81,5.34),(.20,5.34),(.20,3.54),(.35,3.54)])
card(ax,3.78,2.79,3.05,1.51,'Signal-basis adapter',['Train-only basis: rank 128','Existing coefficient adapter',r'Prediction $b$: 2 × 1056'],'teal',7.55)
arrow(ax,[(3.37,3.54),(3.78,3.54)])
card(ax,7.24,2.79,3.48,1.51,'Conditioned residual',[r'$c_0=B^{\mathsf{T}}P_0b$',r'$c=c_0+0.1s\,\tanh F_\psi(z)$','128 corrected coefficients'],'orange',7.9)
arrow(ax,[(6.83,3.54),(7.24,3.54)])
card(ax,11.13,2.79,3.48,1.51,'Clean A/E outputs',[r'$\widehat u=Bc+\mathrm{endpoints}(b)$',r'$\widehat h=W^{-1}\widehat u$','2 × 1056 samples'],'teal',7.7)
arrow(ax,[(10.72,3.54),(11.13,3.54)])
card(ax,.35,.48,6.47,1.63,'Observation-derived conditioning',['262 normalized coordinates + 16 noise features','192 low-frequency continuous-phase features',r'$\log(1+|C|)\{1,\cos\theta,\sin\theta\}$','Inputs: u, b, known mask m, fixed noise calibration'],'gray',7.75)
card(ax,7.24,.48,7.37,1.63,'Residual MLP and preserved baseline',['470 → 192 → two residual blocks → 128','LayerNorm + linear + GELU + dropout + linear',r'$s=\mathrm{RMS}(u)\sqrt{2N}$; zero-initialized output layer',r'$P_0$ removes DC / Nyquist; endpoints of b are retained'],'orange',7.8)
arrow(ax,[(6.82,1.30),(7.02,1.30),(7.02,2.40),(8.14,2.40),(8.14,2.79)],'muted',True)
arrow(ax,[(10.50,2.11),(10.50,2.79)],'orange',True)
ax.text(1.02,2.48,'Known u, m, time coordinate',ha='left',fontsize=7.25,color=C['muted'])
arrow(ax,[(1.45,2.50),(1.45,2.79)],'muted',True)
# A label below the panels explains the arrow grammar in the caption rather than adding decorative elements.
save(fig,'architecture_dgf_20261015')

fig,ax=setup(9.28)
panel(ax,.07,6.14,14.86,3.06,'a  |  Physical data, independent source splits and fixed preprocessing')
card(ax,.35,6.57,3.20,1.83,'Physical ringdown prior',['Source-frame binary prior','Kerr modes + LISA response','Second-generation TDI A/E','Fractional-frequency signals'],'blue',7.9)
card(ax,3.93,6.57,3.20,1.83,'Independent source splits',['Train: 60,000 sources','Validation: 1,500 sources','Test: 6,000 reserved seeds','Generate test after selection'],'gray',7.7)
card(ax,7.51,6.57,3.20,1.83,'Train-only augmentation',['Two variable gaps: 32–52 s','One fixed 33-s gap; new noise','Three views per source','180,000 observations / epoch'],'violet',7.55)
card(ax,11.09,6.57,3.53,1.83,'Whitening + features',['512 noise-only crops → PSD','128 noise-only crops → RMS','CWT / db4; train-only basis',r'Observation: $d=Gh+n$'],'teal',7.45)
for a,b in [(3.55,3.93),(7.13,7.51),(10.71,11.09)]:arrow(ax,[(a,7.49),(b,7.49)])
ax.text(7.5,6.29,'All features come from each observation. Clean targets provide training supervision.',ha='center',fontsize=7.65,color=C['muted'])

panel(ax,.07,3.49,14.86,2.48,'b  |  Staged optimization with the original DGF backbone')
card(ax,.35,3.86,4.05,1.56,'1  Residual-head pilots',['Same 6,000 train sources per variant','Phase / noise / combined features','Up to 60 epochs; patience 20'],'blue',7.65)
card(ax,4.80,3.86,4.43,1.56,'2  Full residual-head training',['60,000 sources; cached DGF outputs','Batch 1024; five-epoch LR warmup','Up to 600 epochs; patience 100'],'orange',7.6)
card(ax,9.63,3.86,4.99,1.56,'3  Selective DGF fine-tuning',['Decoder, refiners, coefficient / frequency heads','422,151 trainable of 9.59M total parameters','Batch 128; up to 200 epochs; patience 50'],'orange',7.55)
arrow(ax,[(4.40,4.63),(4.80,4.63)])
arrow(ax,[(9.23,4.63),(9.63,4.63)])
arrow(ax,[(12.85,6.57),(12.85,6.02),(.20,6.02),(.20,4.63),(.35,4.63)])

panel(ax,.07,.10,14.86,3.17,'c  |  Validation-only decisions, checkpoint freeze and independent testing')
card(ax,.35,.92,6.46,1.77,'Validation gate: all 1,500 sources',['Primary: mean direct joint A/E PSD mismatch','Uniform 32–52 s gaps; fixed 33 / 50 s diagnostics','Raw + EMA (0.995); plateau scheduler; early stopping','Initial checkpoint remains an eligible candidate'],'blue',7.7)
card(ax,7.27,.92,7.35,1.77,'Freeze → generate new test → evaluate',['Freeze model weights, configuration and source hashes','Generate 6,000 new source-disjoint test observations','Compare identical targets / gaps / noise; paired bootstrap','All rows retained; direct signed overlap at fixed time / phase'],'teal',7.6)
arrow(ax,[(6.81,1.79),(7.27,1.79)])
# Validation governs all stages; use one outside-margin feedback route.
arrow(ax,[(.35,1.79),(.20,1.79),(.20,3.68),(2.37,3.68),(2.37,3.86)],'blue',True)
ax.text(7.5,.42,r'Training objective: $\mathcal{M}_{AE}+0.2E_S+0.02E_{\rm time,gap}$   |   Atomic model / EMA / optimizer / scheduler / RNG checkpoints',ha='center',fontsize=7.75,color=C['ink'])
save(fig,'training_workflow_dgf_20261015')
