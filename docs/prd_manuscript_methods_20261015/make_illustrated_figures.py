"""Vector DGF illustrations. All miniature representations are explicitly schematic."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Arc

OUT = Path(__file__).resolve().parent / 'source_edited'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'svg.fonttype': 'none', 'mathtext.fontset': 'dejavusans'})
C = {'ink':'#26384c','muted':'#718096','blue':'#326aa8','teal':'#228b8d',
     'purple':'#8065ae','orange':'#c67b2f','line':'#62758c'}
F = {'blue':'#f0f5fc','teal':'#eef8f6','purple':'#f6f2fa','orange':'#fff6ec','gray':'#f5f7fa'}

def canvas(height):
    fig = plt.figure(figsize=(7.35, 7.35*height/15))
    ax = fig.add_axes([.008,.008,.984,.984])
    ax.set(xlim=(0,15), ylim=(0,height)); ax.axis('off')
    return fig, ax

def text(ax,x,y,s,fs=7.7,color='ink',weight='normal',ha='center',width=None):
    a = ax.text(x,y,s,fontsize=fs,color=C[color],fontweight=weight,ha=ha,va='center',zorder=8)
    if width: a._width = width
    return a

def box(ax,x,y,w,h,title,col='blue',fs=8.4):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=.13',
        facecolor=F[col],edgecolor=C.get(col,'#a4b2c3'),lw=.9,zorder=2))
    text(ax,x+w/2,y+h-.26,title,fs,col if col in C else 'ink','bold',width=w-.22)

def panel(ax,y,h,title,title_offset=.31):
    ax.add_patch(FancyBboxPatch((.07,y),14.86,h,boxstyle='round,pad=.01,rounding_size=.15',
        facecolor='#fcfdff',edgecolor='#dce5ef',lw=.8,zorder=0))
    text(ax,.25,y+h-title_offset,title,9.1,weight='bold',ha='left',width=14.4)

def arrow(ax,points,col='line',dash=False):
    ls=(0,(3,2)) if dash else '-'
    for p,q in zip(points[:-2],points[1:-1]):
        ax.plot([p[0],q[0]],[p[1],q[1]],color=C[col],lw=1.05,ls=ls,zorder=1)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=8,
        color=C[col],lw=1.05,linestyle=ls,shrinkA=0,shrinkB=1,zorder=1))

def wave(ax,x,y,w,h,noise=False,gap=None,color='blue',seed=18):
    t=np.linspace(0,1,200)
    clean=np.exp(-3.3*t)*np.sin(2*np.pi*8*t)+.22*np.exp(-4.5*t)*np.sin(2*np.pi*13*t+.4)
    v=clean.copy()
    if gap:
        mask=(t>gap[0])&(t<gap[1]); v[mask]=0
        ax.add_patch(Rectangle((x+w*gap[0],y-h*.49),w*(gap[1]-gap[0]),h*.98,
            facecolor='#efc1bd',edgecolor='none',alpha=.7,zorder=3))
    if noise: v+=np.random.default_rng(seed).normal(0,.11,len(t))
    ax.plot([x,x+w],[y,y],color='#d4dee9',lw=.6,zorder=3)
    ax.plot(x+w*t,y+h*.42*v,color=C[color],lw=.82,zorder=5)

def heatmap(ax,x,y,w,h):
    t=np.linspace(0,1,33); f=np.linspace(0,1,14)
    T,Fm=np.meshgrid((t[:-1]+t[1:])/2,(f[:-1]+f[1:])/2)
    p=(np.exp(-((Fm-.35)/.12)**2)+.4*np.exp(-((Fm-.7)/.1)**2))*np.exp(-2.8*T)
    p[:,(T[0]>.28)&(T[0]<.39)]*=.15
    ax.pcolormesh(x+w*t,y+h*f,p,cmap='Blues',vmin=0,vmax=1,shading='flat',zorder=4)
    text(ax,x-.13,y+h*.55,'f',6.6,'blue')
    text(ax,x+w+.13,y-.04,'t',6.6,'blue')

def subbands(ax,x,y,w,h):
    t=np.linspace(0,1,180)
    for j,f in enumerate([3,6,11]):
        ax.plot(x+w*t,y+h*(.18+.32*j)+h*.10*np.sin(2*np.pi*f*t)*np.exp(-t),
                color=C['purple'],lw=.72,zorder=5)
    for left,col in [(.22,'#c6b3e1'),(.47,'#9b82c3')]:
        ax.add_patch(Rectangle((x+w*left,y),w*.31,h,facecolor=col,
            edgecolor=C['purple'],alpha=.28,lw=.65,zorder=4))

def nodes(ax,x,y,w,h,color='orange'):
    counts=[3,4,3]
    xx=np.linspace(x,x+w,3)
    yy=[np.linspace(y,y+h,n) for n in counts]
    for j in range(2):
        for a in yy[j]:
            for b in yy[j+1]:
                ax.plot([xx[j],xx[j+1]],[a,b],color=C[color],alpha=.28,lw=.55,zorder=4)
    for j in range(3):
        for a in yy[j]:
            ax.add_patch(Circle((xx[j],a),.06,facecolor=C[color],edgecolor='white',lw=.4,zorder=6))

def matrix(ax,x,y,w,h,rows=6,cols=4,color='teal'):
    for i in range(rows):
        for j in range(cols):
            shade=.20+.64*(.5+.5*np.sin(1.8*i+.9*j))
            ax.add_patch(Rectangle((x+w*j/cols,y+h*i/rows),w/cols*.82,h/rows*.82,
                facecolor=C[color],alpha=shade,edgecolor='none',zorder=4))

def psd(ax,x,y,w,h,color='teal',flat=False):
    t=np.linspace(0,1,70)
    v=.43+.08*np.sin(38*t) if flat else .14+.72*np.exp(-8*t)+.50*t**3
    ax.plot([x,x,x+w],[y+h,y,y],color='#c0cddd',lw=.65,zorder=4)
    ax.plot(x+w*t,y+h*v,color=C[color],lw=1,zorder=5)

def layer_stack(ax,x,y,w,h,trained=False):
    for j in range(3,-1,-1):
        ax.add_patch(FancyBboxPatch((x+.09*j,y+.08*j),w,h,
            boxstyle='round,pad=.01,rounding_size=.07',facecolor=F['teal' if j>1 or not trained else 'orange'],
            edgecolor=C['teal' if j>1 or not trained else 'orange'],lw=.8,zorder=5-j))

def finish(fig,ax,name):
    fig.canvas.draw(); rr=fig.canvas.get_renderer()
    for a in ax.texts:
        if hasattr(a,'_width'):
            limit=ax.transData.transform((a._width,0))[0]-ax.transData.transform((0,0))[0]
            actual=a.get_window_extent(rr).width
            if actual>limit: a.set_fontsize(a.get_fontsize()*limit/actual*.985)
    # Reduce height mildly to retain the two-column paper's existing figure footprint.
    fig.set_size_inches(7.35,fig.get_figheight()*.91)
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/(name+'.'+ext),dpi=200,facecolor='white')
    plt.close(fig)

fig,ax=canvas(11.6)
panel(ax,6.20,5.29,'a  |  DGF: two representations, one shared encoder and learned waveform decoder')
box(ax,.35,7.05,2.15,3.59,'A/E observation','gray')
wave(ax,.57,9.85,1.72,.44,True,(.28,.39),'blue')
wave(ax,.57,9.35,1.72,.44,True,(.28,.39),'teal',28)
text(ax,.48,9.85,'A',6.2,'blue'); text(ax,.48,9.35,'E',6.2,'teal')
text(ax,1.425,8.96,r'$d=Gh+n$',8.1)
text(ax,1.425,8.65,'2 × 1056; 2 Hz',7.3)
arrow(ax,[(1.425,8.43),(1.425,8.21)])
ax.add_patch(FancyBboxPatch((.51,7.28),1.83,.86,boxstyle='round,pad=.01,rounding_size=.09',
    facecolor='white',edgecolor='#b5c4d6',lw=.75,zorder=3))
text(ax,1.425,7.94,'PSD whitening',7.8,weight='bold')
text(ax,1.425,7.67,r'$u=Wd$',8.1)
text(ax,1.425,7.43,'Noise-only PSD',7.0)

box(ax,3.15,8.79,3.40,1.85,'Time-frequency branch')
heatmap(ax,3.48,9.40,2.72,.60)
text(ax,4.85,9.16,'RMS-scaled Morlet CWT',7.5)
text(ax,4.85,8.96,r'$|C|,\arg C$: 2 × 16 × 264',7.5)
box(ax,6.94,8.79,2.51,1.85,'Mixing + adapter',fs=8.0)
nodes(ax,7.30,9.53,1.78,.48,'blue')
text(ax,8.195,9.25,'Time / scale mixing',7.0)
text(ax,8.195,9.03,'Physical-band CWT',7.0)
box(ax,3.15,6.79,6.30,1.59,'Multiresolution temporal branch','purple')
subbands(ax,3.42,7.06,1.77,.72)
arrow(ax,[(5.28,7.42),(5.59,7.42)],'purple')
matrix(ax,5.68,7.09,.69,.65,4,3,'purple')
for yy,s in zip([7.83,7.57,7.30,7.04],['8 signed db4 subbands','32 × 64 patches; stride 32','Overlapping temporal context','Features: 8 × 32 × 64']):
    text(ax,7.99,yy,s,7.35,width=2.65)
arrow(ax,[(2.34,7.73),(2.81,7.73),(2.81,9.70),(3.15,9.70)],'blue')
arrow(ax,[(2.81,7.73),(2.81,7.45),(3.15,7.45)],'purple')
arrow(ax,[(6.55,9.70),(6.94,9.70)],'blue')
ax.add_patch(Circle((10.02,8.75),.19,facecolor='white',edgecolor=C['line'],lw=1,zorder=3))
text(ax,10.02,8.75,'+',12)
arrow(ax,[(9.45,9.70),(10.02,9.70),(10.02,8.94)],'blue')
arrow(ax,[(9.45,7.45),(10.02,7.45),(10.02,8.56)],'purple')
box(ax,10.60,8.20,3.95,2.44,'DGF encoder','teal')
text(ax,12.575,10.02,'Token / conv. / position encoding',7.3,width=3.58)
layer_stack(ax,11.19,8.92,2.54,.68)
text(ax,12.46,9.38,'Channel-wise attention',7.5,'teal',weight='bold',width=2.33)
text(ax,12.46,9.12,'Feedforward + residual',7.1)
text(ax,12.575,8.50,'16 blocks; features 8 × 32 × 64',7.4)
arrow(ax,[(10.21,8.75),(10.60,8.75)])
box(ax,10.60,6.79,3.95,1.13,'Learned waveform decoder','teal')
text(ax,12.575,7.41,'Patch synthesis + overlap average',7.3,width=3.65)
text(ax,12.575,7.08,'Subband sum → Conv1d → rescale',7.25,width=3.65)
arrow(ax,[(12.575,8.20),(12.575,7.92)])
text(ax,7.5,6.45,'Shared A/E encoder weights  |  Feature dimensions per channel; batch omitted',7.35,'muted')

panel(ax,.10,5.85,'b  |  Joint-channel refinement, noise-conditioned coefficients and physical outputs')
box(ax,.35,3.24,3.10,1.98,'Mask + A/E refiners','orange')
for j,n in enumerate([9,5,3]):
    for xx in np.linspace(.78,3.00,n):
        ax.add_patch(Circle((xx,4.21+.23*j),.047,facecolor=C['orange'],alpha=.8,zorder=5))
text(ax,1.90,3.94,'Widths 24 / 32; dilated Conv1d',7.2,width=2.82)
text(ax,1.90,3.57,'Dilations: 1, 2, 4, 8, 16, 32',7.2,width=2.82)
arrow(ax,[(12.575,6.79),(12.575,6.08),(.20,6.08),(.20,4.21),(.35,4.21)])
box(ax,3.86,3.24,3.10,1.98,'Train-derived signal basis','teal')
matrix(ax,4.21,4.08,.98,.72,4,6)
matrix(ax,5.57,4.08,.18,.72,6,1)
text(ax,5.36,4.44,'×',9.5,'teal')
text(ax,6.20,4.44,'→',9.3,'teal')
matrix(ax,6.50,4.20,.16,.48,4,1)
text(ax,5.41,3.94,'Rank 128; existing adapter',7.2)
text(ax,5.41,3.57,r'$c_0=B^{\mathsf{T}}P_0b$',8.1)
arrow(ax,[(3.45,4.21),(3.86,4.21)])
box(ax,7.37,3.24,3.10,1.98,'Conditioned coefficient head','orange',8.1)
nodes(ax,7.91,4.14,2.0,.65)
text(ax,8.92,3.94,'Residual MLP: 470 → 192 → 128',7.0,width=2.8)
text(ax,8.92,3.57,'128 bounded corrections',7.4)
arrow(ax,[(6.96,4.21),(7.37,4.21)])
box(ax,10.88,3.24,3.67,1.98,'Physical A/E outputs','teal')
wave(ax,11.17,4.65,3.06,.39,color='blue')
wave(ax,11.17,4.22,3.06,.39,color='teal')
text(ax,12.715,3.83,r'$\widehat u=Bc+\mathrm{endpoints}(b)$',7.7)
text(ax,12.715,3.51,r'$\widehat h=W^{-1}\widehat u$; 2 × 1056',7.7)
arrow(ax,[(10.47,4.21),(10.88,4.21)])
text(ax,1.65,2.96,'Known u, mask m, time',7.0,'muted')
arrow(ax,[(1.65,2.94),(1.65,3.24)],'muted',True)
box(ax,.35,.48,6.61,2.18,'Observation-derived conditioning','gray')
matrix(ax,.96,1.36,.98,.62,2,7,'blue')
psd(ax,3.26,1.36,1.1,.62)
theta=np.linspace(0,2*np.pi,90)
ax.plot(5.80+.34*np.cos(theta),1.70+.34*np.sin(theta),color=C['orange'],lw=.8,zorder=4)
ax.plot([5.8,6.07],[1.70,1.91],color=C['orange'],lw=1.1,zorder=4)
text(ax,1.45,1.06,'262 coordinates',7.2)
text(ax,3.80,1.06,'16 noise features',7.2,width=1.78)
text(ax,5.80,1.06,'192 phase features',7.2,width=1.94)
text(ax,3.655,.71,'Inputs: u, b, mask m and fixed noise calibration',7.2,width=6.15)
box(ax,7.37,.48,7.18,2.18,'Residual correction preserves the initial model','orange')
text(ax,10.96,1.92,r'$c=c_0+0.1s\,\tanh F_\psi(z)$',10.2)
text(ax,10.96,1.51,'470 → 192 → two residual blocks → 128',7.8)
text(ax,10.96,1.12,r'$s=\mathrm{RMS}(u)\sqrt{2N}$; zero-initialized output layer',7.8,width=6.75)
text(ax,10.96,.74,r'$P_0$ removes DC / Nyquist; endpoints of b are retained',7.45,width=6.75)
arrow(ax,[(6.96,1.67),(7.16,1.67),(7.16,2.91),(8.92,2.91),(8.92,3.24)],'muted',True)
arrow(ax,[(11.28,2.66),(11.28,2.91),(9.58,2.91),(9.58,3.24)],'orange',True)
finish(fig,ax,'architecture_dgf_20261015')

fig,ax=canvas(10.70)
panel(ax,6.69,3.90,'a  |  Physical signals, independent source populations and observation-derived features')
box(ax,.35,7.48,3.20,2.60,'Physical ringdown + LISA','blue')
wave(ax,.65,9.30,2.60,.68)
text(ax,1.95,8.75,'Source-frame binary prior',7.6)
text(ax,1.95,8.43,'Kerr modes + finite-arm response',7.3,width=2.92)
text(ax,1.95,8.11,'Second-generation TDI A/E',7.3)
text(ax,1.95,7.77,'Fractional-frequency signals',7.3)
box(ax,3.93,7.48,3.20,2.60,'Source-disjoint splits','gray')
for y,s,col in [(9.26,'Train: 60,000','blue'),(8.78,'Validation: 1,500','purple'),(8.30,'Test: 6,000','teal')]:
    matrix(ax,4.17,y-.12,.49,.23,2,4,col)
    text(ax,5.73,y,s,7.5,width=2.03)
text(ax,5.53,7.78,'Generate test after selection',7.05,width=2.92)
box(ax,7.51,7.48,3.20,2.60,'Three train-only views','purple')
for i,(yy,g,s) in enumerate([(9.30,(.28,.345),'variable'),(8.80,(.265,.370),'variable'),(8.30,(.28,.346),'33 s; new noise')]):
    wave(ax,7.72,yy,1.38,.32,True,g,'purple',30+i)
    text(ax,9.83,yy,s,6.9,width=1.48)
text(ax,9.11,7.78,'180,000 observations / epoch',7.05,width=2.92)
box(ax,11.09,7.48,3.53,2.60,'Fixed whitening + features','teal',8.2)
psd(ax,11.44,9.02,1.04,.47)
psd(ax,13.17,9.02,1.04,.47,flat=True)
arrow(ax,[(12.65,9.26),(12.99,9.26)],'teal')
text(ax,11.96,8.83,'PSD',6.8,'teal'); text(ax,13.69,8.83,'Whitened',6.8,'teal')
text(ax,12.855,8.51,'512 noise-only crops → PSD',7.15)
text(ax,12.855,8.18,'First 128 → expected noise RMS',7.15,width=3.2)
text(ax,12.855,7.78,'CWT / db4; train-only signal basis',7.0,width=3.2)
for p,q in [(3.55,3.93),(7.13,7.51),(10.71,11.09)]:
    arrow(ax,[(p,8.63),(q,8.63)])
text(ax,7.5,7.01,'Observation d = Gh + n  |  Features follow each observation; clean targets supervise training',7.3,'muted')

panel(ax,3.82,2.70,'b  |  Staged optimization with the original DGF backbone')
box(ax,.35,4.20,4.05,1.75,'1  Residual-head pilots','blue')
nodes(ax,.65,4.73,.94,.55,'blue')
for yy,s in zip([5.18,4.89,4.57],['6,000 sources; 3 variants','Phase / noise / combined','Max 60; patience 20']):
    text(ax,3.00,yy,s,7.3,width=2.28)
box(ax,4.80,4.20,4.43,1.75,'2  Full residual-head training','orange')
nodes(ax,5.07,4.73,.94,.55)
for yy,s in zip([5.18,4.89,4.57],['60,000 sources; batch 1024','Frozen backbone; new head','Max 600; patience 100']):
    text(ax,7.66,yy,s,7.3,width=2.78)
box(ax,9.63,4.20,4.99,1.75,'3  Selective DGF fine-tuning','orange')
layer_stack(ax,9.86,4.68,.77,.47,True)
for yy,s in zip([5.18,4.89,4.57],['422,151 trainable / 9.59M total','Decoder, refiners and adapters','Batch 128; max 200; patience 50']):
    text(ax,12.77,yy,s,7.3,width=3.46)
arrow(ax,[(4.40,5.07),(4.80,5.07)])
arrow(ax,[(9.23,5.07),(9.63,5.07)])
arrow(ax,[(12.855,7.48),(12.855,6.61),(.20,6.61),(.20,5.07),(.35,5.07)])

panel(ax,.10,3.55,'c  |  Validation decisions, checkpoint freeze and new independent testing',.24)
box(ax,.35,.92,6.46,2.18,'Validation: all 1,500 sources','blue')
t=np.linspace(0,1,100)
ax.plot([.70,.70,2.34],[2.40,1.71,1.71],color='#c2cfdf',lw=.7,zorder=4)
ax.plot(.70+1.64*t,1.81+.51*np.exp(-5*t)+.016*np.sin(42*t),color=C['blue'],lw=1.05,zorder=5)
ax.plot([2.10,2.10],[1.71,2.4],ls=(0,(2,2)),color=C['orange'],lw=.85,zorder=5)
text(ax,1.52,1.50,'Plateau + early stop',7.0,'blue')
for yy,s in zip([2.47,2.12,1.77],['Primary: mean direct A/E mismatch','Raw + EMA; learning-rate schedule','Initial checkpoint remains eligible']):
    text(ax,4.53,yy,s,7.3,width=4.17)
text(ax,3.58,1.17,'Variable 32–52 s; fixed 33 / 50 s diagnostics',7.5)
box(ax,7.27,.92,7.35,2.18,'Freeze → generate new test → evaluate','teal')
# Model lock, fresh-source grid and paired comparison form a miniature test illustration.
ax.add_patch(Arc((7.95,2.30),.42,.46,theta1=0,theta2=180,color=C['teal'],lw=1.25,zorder=5))
ax.add_patch(FancyBboxPatch((7.62,1.80),.66,.54,boxstyle='round,pad=.01,rounding_size=.06',
    facecolor='#d2e9e5',edgecolor=C['teal'],lw=.9,zorder=5))
ax.add_patch(Circle((7.95,2.09),.043,facecolor=C['teal'],zorder=6))
matrix(ax,8.56,1.80,.60,.57,3,4)
for j,hh in enumerate([.21,.46,.30,.53]):
    ax.add_patch(Rectangle((9.38+j*.16,1.80),.10,hh,facecolor=C['teal'],alpha=.65,zorder=5))
for yy,s in zip([2.47,2.12,1.77],['6,000 new source-disjoint tests','Shared targets, gaps and noise','Paired bootstrap; all rows retained']):
    text(ax,12.35,yy,s,7.3,width=4.29)
text(ax,10.945,1.17,'Direct signed overlap at fixed time and phase',7.5)
arrow(ax,[(6.81,2.00),(7.27,2.00)])
arrow(ax,[(12.12,4.20),(12.12,3.73),(14.80,3.73),(14.80,3.20),(5.60,3.20),(5.60,3.10)])
arrow(ax,[(.35,2.00),(.13,2.00),(.13,4.04),(2.37,4.04),(2.37,4.20)],'blue',True)
text(ax,7.5,.42,r'Objective: $\mathcal{M}_{AE}+0.2E_S+0.02E_{\rm time,gap}$  |  Atomic model / EMA / optimizer / scheduler / RNG checkpoints',7.35)
finish(fig,ax,'training_workflow_dgf_20261015')
