from pathlib import Path
import json,hashlib,zipfile
P=Path(__file__).resolve().parent;S=P/'source';p=S/'DGF.tex';s=p.read_text(encoding='utf-8')
start=s.index('Figure~\\ref{fig:examples} shows');end=s.index('\n\\begin{figure*}',start)
s=s[:start]+r'''Figure~\ref{fig:examples} shows three $\rho\ge10$ sources with dominant-mode frequencies of 15.4, 16.5 and 31.5 mHz. Their 37-, 35- and 33-s gaps contain 26.4\%, 17.2\% and 23.6\% of the joint signal squared norms, respectively. The direct noise-weighted mismatches are $7.41\times10^{-4}$, $7.88\times10^{-4}$ and $2.04\times10^{-3}$; gap-local relative errors are 4.2\%, 5.5\% and 6.9\%. Both the full observations and expanded waveform views retain the noise. Source 3 has a dominant-mode period of 31.7 s and a damping time of 32.1 s, resolving more oscillations within a shorter interval.'''+s[end:]
start=s.index('\\caption{Waveform reconstruction for');end=s.index('\\label{fig:examples}',start)
s=s[:start]+r'''\caption{Waveform reconstruction for three sources in the $A$ (left) and $E$ (right) channels. Each labeled panel shows the full 528-s observation above an amplitude and time zoom. Both views include $d=Gh+n$ (gray), the clean signal $h$, gapped signal $Gh$ and DGF reconstruction $\widehat h$. Every ordinate gives fractional frequency in units of $10^{-18}$. Shading marks gaps of 37, 35 and 33 s, respectively. Curves connect the original 0.5-s samples; the zoom axes clip noise excursions outside their limits. Source seeds are 2042203874, 2042204884 and 2042204076. Appendix~\ref{app:model_analysis} gives the example criteria.}'''+s[end:]
s=s.replace(r'\label{fig:examples}\par\vspace{8pt}',r'\label{fig:examples}\end{figure*}'+'\n'+r'\begin{figure*}[tp]\centering')
s=s.replace('source 2042202428','source 2042203874')
start=s.index('Figure~\\ref{fig:tdi_log_amplitudes} plots');end=s.index(' Figure~\\ref{fig:psd}',start)
s=s[:start]+r'''Figure~\ref{fig:tdi_log_amplitudes} follows the waveform absolute values through three dominant-mode damping times after the joint clean-envelope peak. This interval resolves the principal decay and zero crossings; Fig.~\ref{fig:examples} shows the full observations. All reported errors use the complete 528-s records.'''+s[end:]
start=s.index('\\caption{Absolute fractional-frequency amplitudes');end=s.index('\\label{fig:tdi_log_amplitudes}',start)
s=s[:start]+r'''\caption{Absolute fractional-frequency amplitudes of the observation (gray), clean signal and DGF reconstruction for the three sources in Fig.~\ref{fig:examples}. Each interval runs from $\max(0,t_{\rm pk}-1.8\tau_{220})$ to $\min(527.5\,\mathrm{s},t_{\rm pk}+3\tau_{220})$, rounded to the preceding native sample. Here $t_{\rm pk}$ is the joint clean analytic-envelope peak. The resulting end times are 255.5, 245.0 and 162.0 s. This source-defined window displays the principal decay in both channels. Shading marks the gap.}'''+s[end:]
s=s.replace(r'\label{fig:tdi_log_amplitudes}'+'\n'+r'\par\vspace{8pt}',r'\label{fig:tdi_log_amplitudes}\end{figure*}'+'\n'+r'\begin{figure*}[tp]\centering')
start=s.index('The examples in Fig.~\\ref{fig:examples}');end=s.index('\n',start)
s=s[:start]+r'''The three examples in Fig.~\ref{fig:examples} span dominant-mode frequencies of 15.4--31.5 mHz and have resolved oscillations, $\rho\ge10$, gap-local relative errors below 7\% and channel peak-phase errors below $2.9^\circ$. Their seeds identify the displayed cases; population statistics include every test source. The display interval in Fig.~\ref{fig:tdi_log_amplitudes} is determined by the clean signal and its injected damping time.'''+s[end:]
p.write_text(s,encoding='utf-8')
# Preserve unchanged PDF files byte-for-byte from the verified cloud source.
z=zipfile.ZipFile(P.parent/'ringdown_shape_audit_20261015/overleaf_final_source.zip')
for n in ['latest_mismatch.pdf','latest_snr_gap.pdf','latest_residual_psd.pdf']:(S/n).write_bytes(z.read(n))
p=S/'reproducibility_manifest.json';v=json.loads(p.read_text());old=v['figure_examples'];sel=json.loads((P/'selected_examples.json').read_text());v['figure_examples']={k:old[k] for k in ['checkpoint_sha256','source_data_sha256'] if k in old};v['figure_examples'].update(sel)
for n in v['figure_sha256']:
 if (S/n).exists():v['figure_sha256'][n]=hashlib.sha256((S/n).read_bytes()).hexdigest()
p.write_text(json.dumps(v,indent=2)+'\n')
