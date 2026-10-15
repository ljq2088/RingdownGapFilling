from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
P=Path(__file__).resolve().parent
a=np.load(P/'population_spectra.npz');f=a['frequency'];valid=(f>0)&(f<1)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(7.35,2.8),layout='constrained')
for c,ax in enumerate(axs):
    ax.loglog(f[valid],a['target'][c,valid],color='#26384c',lw=1.05,ls=(0,(4,2)))
    ax.loglog(f[valid],a['noise'][c,valid],color='#8a97aa',lw=.9)
    ax.loglog(f[valid],a['residual'][c,valid],color='#c76625',lw=1.1)
    ax.set(xlim=(f[1],1),xlabel='Frequency (Hz)',ylabel=r'PSD (Hz$^{-1}$)' if c==0 else '')
    ax.set_title(f'({"ab"[c]}) {"AE"[c]} channel',loc='left',fontsize=8.5)
    ax.grid(alpha=.15,lw=.5,which='major')
fig.legend(handles=[Line2D([],[],color='#26384c',lw=1,ls=(0,(4,2)),label='Target signal'),Line2D([],[],color='#8a97aa',lw=1,label='Instrument noise'),Line2D([],[],color='#c76625',lw=1.1,label='Reconstruction residual')],loc='outside upper center',ncols=3,frameon=False,fontsize=8,handlelength=2.6)
for ext in ('pdf','svg','png'):fig.savefig(P/'source'/('latest_residual_psd.'+ext),dpi=200,facecolor='white')
plt.close(fig)
