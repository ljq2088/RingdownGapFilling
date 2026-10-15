from pathlib import Path
import json,hashlib,shutil
P=Path(__file__).resolve().parent;S=P/'source';f=S/'DGF.tex';s=f.read_text(encoding='utf-8')
s=s.replace('For the three sources in Fig.~\\ref{fig:examples}, we fix','For each illustrated source, we fix')
text=r'''
Figure~\ref{fig:conditional_frequency} compares two additional sources at 7.3 and 38.7 mHz on a common observer-frame time axis. Their damping times are 135.6 and 26.7 s, respectively. This representation resolves the difference in oscillation period that is partly absorbed by the $t/\tau_{220}$ scaling in Fig.~\ref{fig:conditional_strain}. The direct $A/E$ mismatches over the full records are $2.60\times10^{-4}$ and $2.63\times10^{-3}$. Over the displayed 240 s, the conditional strain errors are 1.24\% and 5.35\%, respectively.

\begin{figure*}[p]\centering
\includegraphics[width=\textwidth]{conditional_frequency_contrast.pdf}
\caption{Conditional strain reconstruction at lower (7.3 mHz, left) and higher (38.7 mHz, right) dominant-mode frequencies. The upper rows show the complete 528-s $A/E$ observations in fractional-frequency units of $10^{-18}$, with the signal-suppression intervals shaded. The lower rows compare injected and conditionally reconstructed strain, together with the projected noise realization, using the colors and projection of Fig.~\ref{fig:conditional_strain}. Both columns use the same observer-frame source-time interval, measured in seconds from the mode-amplitude reference epoch. Strain units are $10^{-18}$ on the left and $10^{-19}$ on the right. Curves connect the native 0.5-s samples, and noise retains its original amplitude. The source seeds are 2042203322 and 2042205712.}\label{fig:conditional_frequency}
\end{figure*}

'''
assert r'\label{fig:conditional_frequency}' not in s
s=s.replace(r'\section{Discussion and Conclusions}',text+r'\section{Discussion and Conclusions}')
needle='The display interval in Fig.~\\ref{fig:tdi_log_amplitudes} is determined by the clean signal and its injected damping time.'
assert needle in s
s=s.replace(needle,needle+r' The two additional examples in Fig.~\ref{fig:conditional_frequency} have $\rho\ge10$, direct mismatch below $5\times10^{-3}$ and gap-local relative error below 15\%; their conditional strain errors are below 6\%. They illustrate recovery at separated frequencies within the test population.')
f.write_text(s,encoding='utf-8')
shutil.copyfile(P/'conditional_frequency_contrast.pdf',S/'conditional_frequency_contrast.pdf')
shutil.copyfile(P/'conditional_source_strain.pdf',S/'conditional_source_strain.pdf')
f=S/'reproducibility_manifest.json';v=json.loads(f.read_text());v['conditional_frequency_contrast']=json.loads((P/'frequency_contrast_report.json').read_text());v['conditional_source_reconstruction']=json.loads((P/'conditional_source_report.json').read_text())
for name in ['conditional_source_strain.pdf','conditional_frequency_contrast.pdf']:v['figure_sha256'][name]=hashlib.sha256((S/name).read_bytes()).hexdigest()
f.write_text(json.dumps(v,indent=2)+'\n')
