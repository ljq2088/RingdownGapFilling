"""Vector DGF illustrations. All miniature representations are explicitly schematic."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Arc

OUT = Path(__file__).resolve().parent / 'source'
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

