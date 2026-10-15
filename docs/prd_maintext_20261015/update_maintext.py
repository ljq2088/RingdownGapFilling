from pathlib import Path
import zipfile,json,shutil
P=Path(__file__).resolve().parent
with zipfile.ZipFile(Path.home()/'Downloads/RingdownGapFilling___PRD_revision_2026_10_09 (56).zip') as z:z.extractall(P/'source')
f=P/'source/DGF.tex';s=f.read_text(encoding='utf-8')
start=s.index(r'\subsection{Conditional reconstruction of incident strain}')
end=s.index(r'\section{Discussion and Conclusions}')
section=s[start:end]
section=section.replace(r'\begin{figure*}[p]',r'\begin{figure*}[!tp]')
section=section.replace('Figure~\\ref{fig:conditional_strain} displays the original observations together with the conditional strain estimates.','Figure~\\ref{fig:conditional_strain} connects the reconstructed detector channels to the incident polarizations for the three examples in Fig.~\\ref{fig:examples}. Each upper panel overlays the noisy observation and direct channel reconstruction, while the lower panels resolve the corresponding damped strain oscillations.')
section=section.replace('Figure~\\ref{fig:conditional_frequency} compares two additional sources at 7.3 and 38.7 mHz on a common observer-frame time axis.','Figure~\\ref{fig:conditional_frequency} extends this comparison to two additional sources at 7.3 and 38.7 mHz on a common observer-frame time axis. Both have $\\rho\\ge10$, direct mismatch below $5\\times10^{-3}$ and gap-local relative error below 15\\%.')
# Keep all result figures before the discussion, retaining the two-column result text flow.
barrier='\\onecolumngrid\n\\clearpage\n\\twocolumngrid\n'
s=s[:start]+section+barrier+s[end:]
needle='The waveform comparisons show recovery of the oscillations across the gaps and their subsequent decay, while gap-local and spectral errors quantify amplitude-sensitive deviations.'
assert needle in s
s=s.replace(needle,needle+' The conditional strain reconstructions in Sec.~\\ref{subsec:incident_strain} connect these channel outputs to the incident polarizations at fixed injected source parameters and resolve the different oscillation and decay scales across the illustrated frequencies.')
old='The two additional examples in Fig.~\\ref{fig:conditional_frequency} have $\\rho\\ge10$, direct mismatch below $5\\times10^{-3}$ and gap-local relative error below 15\\%; their conditional strain errors are below 6\\%. They illustrate recovery at separated frequencies within the test population.'
assert old in s
s=s.replace(old,'The source seeds in Figs.~\\ref{fig:conditional_strain} and \\ref{fig:conditional_frequency} identify the incident-strain examples analyzed in Sec.~\\ref{subsec:incident_strain}; their selection and display intervals are specified there.')
f.write_text(s,encoding='utf-8',newline='\n')
f=P/'source/reproducibility_manifest.json';v=json.loads(f.read_text(encoding='utf-8'));v['manuscript_layout']={'change':'Conditional strain figures and frequency comparison placed within Reconstruction results, before Discussion and Conclusions. Related discussion, captions and appendix references synchronized.','figure_data':'unchanged','evaluation_statistics':'unchanged'};f.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8',newline='\n')
script=(P.parent/'ringdown_source_overlay_20261015/stage_compile.py').read_text(encoding='utf-8').replace("'ringdown_source_noise_20261015'","'ringdown_maintext_20261015'");(P/'stage_compile.py').write_text(script,encoding='utf-8')
