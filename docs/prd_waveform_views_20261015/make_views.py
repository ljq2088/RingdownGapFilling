from pathlib import Path
import json, hashlib, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter, MaxNLocator

HERE=Path(__file__).resolve().parent
OUT=HERE/'source'
BASE=HERE.parent
native=np.load(BASE/'figure4_oscillatory_20261015/figure4_native_samples.npz')
selection=json.loads((BASE/'optimization_round_20261015/selected_oscillatory_examples.json').read_text())['selected']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
 'axes.titlesize':9,'legend.fontsize':8,'pdf.fonttype':42,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False,'path.simplify':False})
C={'plus':'#1e334f','cross':'#14847b','pred':'#cc5f20','shade':'#e6edf5','amp':'#929dab'}
def save(fig,name):
    for ext in ('pdf','svg','png'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=240,facecolor='white')
    plt.close(fig)
def style(ax):
    ax.tick_params(labelsize=7,length=3,pad=2)
    ax.grid(axis='y',alpha=.17,lw=.5)
def positive(x):
    return np.where(x>0,x,np.nan)

fig,axes=plt.subplots(3,2,figsize=(7.35,6.8),layout='constrained')
fig.set_constrained_layout_pads(w_pad=.035,h_pad=.04,hspace=.075,wspace=.04)
audit=[]
for row,s in enumerate(selection):
    seed=int(native['seeds'][row]);assert seed==s['seed']
    tracepath=BASE/f'optimization_round_20261015/waveform_trace_oscillatory/trace_{seed}.npz'
    z=np.load(tracepath);t=z['time_ssb'];strain=z['source'];tau=s['tau_220']
    ix=(t>=0)&(t<=6*tau);tt=t[ix];h=strain[ix]
    scale=float(abs(h).max());assert scale>0
    # The source arrays are the same polarizations supplied to the response.
    crop=int(z['crop_start']);assert np.array_equal(z['tdi'][:,crop:crop+1056].astype('f4'),native['truth'][row].astype('f4'))
    a,b=axes[row]
    a.plot(tt,h.real/scale,color=C['plus'],lw=1.2)
    a.plot(tt,h.imag/scale,color=C['cross'],ls=(0,(4,2)),lw=1.05)
    a.axhline(0,color='#aab3bf',lw=.5,zorder=0)
    a.set(xlim=(0,6*tau),ylim=(-1.08,1.08),ylabel=r'$h_{+,\times}/H_{\rm src}$')
    a.set_title(f'({chr(97+2*row)}) Source {row+1} | signed strain',loc='left',pad=6)
    b.semilogy(tt,positive(abs(h.real)/scale),color=C['plus'],lw=1.15)
    b.semilogy(tt,positive(abs(h.imag)/scale),color=C['cross'],ls=(0,(4,2)),lw=1.0)
    b.semilogy(tt,positive(abs(h)/scale),color=C['amp'],ls=':',lw=1)
    b.set(xlim=(0,6*tau),ylim=(1e-4,1.3),ylabel=r'$|h_{+,\times}|/H_{\rm src}$')
    b.set_title(f'({chr(98+2*row)}) Source {row+1} | absolute strain',loc='left',pad=6)
    b.yaxis.set_major_locator(LogLocator(base=10,numticks=5));b.yaxis.set_minor_formatter(NullFormatter())
    for ax in (a,b):
        style(ax);ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel(r'$t-t_{\rm ref}$ (s)' if row==2 else '')
    audit.append({'seed':seed,'source_scale_strain':scale,'tau_220_seconds':tau,'source_plot_time_seconds':[float(tt[0]),float(tt[-1])],
                  'source_is_simulator_input':True,'stored_tdi_crop_reproduced':True,'trace_sha256':hashlib.sha256(tracepath.read_bytes()).hexdigest()})
fig.legend(handles=[Line2D([],[],color=C['plus'],lw=1.3,label=r'$h_+$'),
                    Line2D([],[],color=C['cross'],ls=(0,(4,2)),lw=1.1,label=r'$h_\times$'),
                    Line2D([],[],color=C['amp'],ls=':',lw=1,label=r'$\sqrt{h_+^2+h_\times^2}$ (right)')],
           loc='outside upper center',ncols=3,frameon=False,handlelength=2.5,columnspacing=2)
save(fig,'source_polarizations_20261015')

fig,axes=plt.subplots(3,2,figsize=(7.35,6.8),layout='constrained')
fig.set_constrained_layout_pads(w_pad=.035,h_pad=.04,hspace=.075,wspace=.04)
t=native['time']
for row,s in enumerate(selection):
    h=native['truth'][row];p=native['reconstruction'][row];mask=native['mask'][row];scale=float(abs(h).max())
    idx=np.flatnonzero(mask);start=t[idx[0]];end=start+.5*len(idx)
    for c,ax in enumerate(axes[row]):
        ax.axvspan(start,end,color=C['shade'],lw=0,zorder=0)
        ax.semilogy(t,positive(abs(h[c])/scale),color=C['plus'],lw=1.15)
        ax.semilogy(t,positive(abs(p[c])/scale),color=C['pred'],lw=1.05,ls=(0,(4,2)))
        ax.set(xlim=(0,528),ylim=(1e-5,1.5),xticks=[0,100,200,300,400,528],
               ylabel=r'$|h_c|/H_{\rm TDI}$' if c==0 else '',xlabel='Time (s)' if row==2 else '')
        ax.set_title(f'({chr(97+2*row+c)}) Source {row+1} | {"AE"[c]} channel',loc='left',pad=6)
        ax.yaxis.set_major_locator(LogLocator(base=10,numticks=6));ax.yaxis.set_minor_formatter(NullFormatter());style(ax)
fig.legend(handles=[Line2D([],[],color=C['plus'],lw=1.3,label=r'Clean signal $|h_c|$'),
                    Line2D([],[],color=C['pred'],ls=(0,(4,2)),lw=1.1,label=r'DGF reconstruction $|\widehat h_c|$'),
                    Patch(facecolor=C['shade'],edgecolor='none',label='Gap')],
           loc='outside upper center',ncols=3,frameon=False,handlelength=2.5,columnspacing=1.5)
save(fig,'tdi_log_amplitudes_20261015')

texpath=OUT/'DGF.tex';tex=texpath.read_text(encoding='utf-8')
anchor='\\section{Fixed-duration gaps and reference comparison}'
assert tex.count(anchor)==1
addition=r'''
\subsection{Source polarizations and logarithmic amplitude}
Figure~\ref{fig:source_polarizations} shows the source polarizations that generate the three responses in Fig.~\ref{fig:examples}. Their signed strain displays the decaying oscillations before the LISA response is applied. The source time origin is the ringdown reference epoch, and each panel extends over six dominant-mode damping times. The source strains enter the forward simulation; DGF estimates the fractional-frequency $A/E$ waveforms. Source-strain inference from the channel estimates requires an inverse-response analysis with specified source geometry and signal history.
\begin{figure*}[t]\centering\includegraphics[width=\textwidth]{source_polarizations_20261015.pdf}
\caption{Source polarizations for the same sources as Fig.~\ref{fig:examples}, in the same row order. Left: signed $h_+$ and $h_\times$ in the polarization basis supplied to the LISA response. Right: their absolute values on a logarithmic scale, together with the polarization amplitude $\sqrt{h_+^2+h_\times^2}$. Each row uses $H_{\rm src}=\max_{0\le t-t_{\rm ref}\le6\tau_{220}}\sqrt{h_+^2+h_\times^2}$ for all curves. The plotted interval contains the post-reference ringdown segment of the simulated source. Curves connect the native 0.5-s samples.}\label{fig:source_polarizations}
\end{figure*}

Figure~\ref{fig:tdi_log_amplitudes} compares the absolute clean and reconstructed channel amplitudes on a logarithmic scale over the full analysis window. This view resolves the decay and the residuals at small amplitudes. The minima follow the zero crossings of the signed waveforms. The signed comparisons in Fig.~\ref{fig:examples} preserve the accompanying phase information.
\begin{figure*}[t]\centering\includegraphics[width=\textwidth]{tdi_log_amplitudes_20261015.pdf}
\caption{Absolute amplitudes of the clean and reconstructed $A$ (left) and $E$ (right) waveforms for the sources in Fig.~\ref{fig:examples}. Each row uses the common scale $H_{\rm TDI}=\max_{c,j}|h_{c,j}|$ from the clean signal. Shading marks the gap. The logarithmic axis resolves small-amplitude structure and differences near zero crossings. All curves use the same physical-time arrays as Fig.~\ref{fig:examples}, with native sampling and fixed arrival time and phase.}\label{fig:tdi_log_amplitudes}
\end{figure*}

'''
tex=tex.replace(anchor,addition+anchor)
cross='The windows extend for 4.3--7.3 dominant-mode damping times after the response peak.'
assert tex.count(cross)==1
tex=tex.replace(cross,cross+' Source-polarization and logarithmic-amplitude views are given in Appendix~\\ref{app:complementary}.')
assert 'Selected successful' not in tex and 'successful illustrations' not in tex
texpath.write_text(tex,encoding='utf-8',newline='\n')
report={'source_cases':audit,'tdi_arrays_sha256':hashlib.sha256((BASE/'figure4_oscillatory_20261015/figure4_native_samples.npz').read_bytes()).hexdigest(),
        'native_dt_seconds':.5,'smoothing':False,'interpolation':False,'source_is_simulator_input':True,
        'source_strain_reconstruction_claimed':False,'population_statistics_unchanged':True,
        'normalization':'Common source-polarization norm for source plots; common two-channel clean peak for TDI plots.'}
(HERE/'waveform_views_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
