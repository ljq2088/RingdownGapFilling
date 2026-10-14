from pathlib import Path
import zipfile,shutil,re,difflib,json
here=Path(__file__).resolve().parent
backup=Path.home()/'Downloads/RingdownGapFilling___PRD_revision_2026_10_09 (24).zip'
shutil.copy2(backup,here/'overleaf_before.zip')
with zipfile.ZipFile(backup) as z:z.extractall(here/'source');old=z.read('DGF.tex').decode('utf-8')
assert old==(here.parent/'figure4_noise_overlay_20261015/source/DGF.tex').read_text()
tex=old
begin=tex.index(r'\section{Additional waveform diagnostics}');finish=tex.index(r'\section{Fixed-duration gaps and reference comparison}',begin)
block=tex[begin:finish]
def extract_figure(label):
 pos=block.index('\\label{'+label+'}');start=block.rfind(r'\begin{figure*}',0,pos);end=block.index(r'\end{figure*}',pos)+len(r'\end{figure*}')
 return block[start:end]
snr_figure=extract_figure('fig:snr_gap')
snr_figure=snr_figure.replace('Additional diagnostics for the same independent test population.','Dependence of reconstruction accuracy on network SNR and gap location for the independent test population.')
log_figure=extract_figure('fig:tdi_log_amplitudes')
metadata=block[block.index('The two sources in Fig.'):block.index('\n\n',block.index('The two sources in Fig.'))]
conventions=block[block.index('The phase statistics in Sec.'):block.index('\n\n',block.index('The phase statistics in Sec.'))]
# Keep the detailed numerical conventions in the reproducibility appendix.
tex=tex[:begin]+metadata+'\n\n'+conventions+'\n\n'+tex[finish:]
anchor=r'Figure~\ref{fig:examples} compares the clean and reconstructed waveforms'
assert tex.count(anchor)==1
tex=tex.replace(anchor,snr_figure+'\n\n'+anchor)
tex=tex.replace(r' Logarithmic-amplitude views are given in Appendix~\ref{app:complementary}.','')
log_paragraph=r'Figures~\ref{fig:tdi_log_amplitudes} and~\ref{fig:psd} show the logarithmic waveform amplitudes and population-averaged residual spectra, respectively. The amplitude panels use the sources in Fig.~\ref{fig:examples} and resolve the decaying tail and the waveform zero crossings.'
anchor=r'Figure~\ref{fig:psd} compares the mean one-sided periodograms'
assert tex.count(anchor)==1
tex=tex.replace(anchor,log_paragraph+'\n'+log_figure+'\n\n'+anchor)
tex=tex.replace('These conditional distributions quantify reconstruction when the strongest part of the signal is suppressed, while retaining the source and noise variation within each group.','Both subsets retain variation in source parameters and noise realizations.')
old_psd=r'Figure~\ref{fig:psd} compares the mean one-sided periodograms of the reconstruction residual and simulated instrument noise in each channel. Both spectra average all 6,000 test sources and are expressed in fractional-frequency units. They show the frequency dependence of the residual after joint denoising and reconstruction. Mode-resolved frequency and damping-time estimation requires fitting the individual ringdown components.'
assert old_psd in tex;tex=tex.replace(old_psd,'')
# Put phase recovery next to the signed waveform comparison, followed by
# amplitude/spectral diagnostics. Slightly reduce the two adjacent figures
# so they share a page without changing any plotted data.
phase_start=tex.index(r'\subsection{Phase recovery}')
phase_end=tex.index(r'\section{Discussion and Conclusions}',phase_start)
phase_block=tex[phase_start:phase_end]
tex=tex[:phase_start]+tex[phase_end:]
anchor=r'\subsection{Amplitude and spectral fidelity}'
tex=tex.replace(anchor,phase_block+anchor)
for name in ['latest_examples.pdf','latest_phase.pdf']:
 tex=tex.replace(r'\includegraphics[width=\textwidth]{'+name+'}',r'\includegraphics[width=0.885\textwidth]{'+name+'}')
def grouped_figures(first_label,second_label):
 global tex
 def locate(label):
  pos=tex.index('\\label{'+label+'}');start=tex.rfind(r'\begin{figure*}',0,pos);end=tex.index(r'\end{figure*}',pos)+len(r'\end{figure*}')
  return tex[start:end]
 first=locate(first_label);second=locate(second_label)
 second_body=second[second.index(r'\centering')+len(r'\centering'):second.rindex(r'\end{figure*}')]
 merged=first.replace(r'\end{figure*}',r'\par\vspace{8pt}'+'\n'+second_body+r'\end{figure*}').replace(r'\begin{figure*}[t]',r'\begin{figure*}[tp]')
 tex=tex.replace(second,'').replace(first,merged)
grouped_figures('fig:examples','fig:phase')
grouped_figures('fig:tdi_log_amplitudes','fig:psd')
# Flush every result figure before the appendices; the discussion can fill
# otherwise unused body-text space while the two-column floats are placed.
tex=tex.replace(r'\appendix',r'\clearpage'+'\n'+r'\appendix')
assert 'app:complementary' not in tex
assert tex.index(r'\label{fig:snr_gap}')<tex.index(r'\label{fig:examples}')<tex.index(r'\label{fig:phase}')<tex.index(r'\label{fig:tdi_log_amplitudes}')<tex.index(r'\label{fig:psd}')<tex.index(r'\appendix')
(here/'source/DGF.tex').write_text(tex,encoding='utf-8',newline='\n')
(here/'manuscript.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),tex.splitlines(True),fromfile='before/DGF.tex',tofile='after/DGF.tex')))
compile_script=(here.parent/'figure4_noise_overlay_20261015/stage_compile.py').read_text().replace('ringdown_noise_overlay_20261015','ringdown_main_figures_20261015')
(here/'stage_compile.py').write_text(compile_script)
print('Moved SNR/gap diagnostics and logarithmic waveform amplitudes into Results. Figure PDFs and numerical results unchanged.')
