from pathlib import Path
import zipfile,re,json
P=Path(__file__).resolve().parent
with zipfile.ZipFile(Path.home()/'Downloads/RingdownGapFilling___PRD_revision_2026_10_09 (58).zip') as z:z.extractall(P/'source')
f=P/'source/DGF.tex';s=f.read_text(encoding='utf-8');original=s
def remove_figure(label):
 global s
 pattern=r'\\begin\{figure\*\}.*?\\end\{figure\*\}'
 found=[m for m in re.finditer(pattern,s,re.S) if r'\label{'+label+'}' in m.group()]
 assert len(found)==1,label
 m=found[0];s=s[:m.start()]+s[m.end():]
remove_figure('fig:tdi_log_amplitudes')
remove_figure('fig:conditional_strain')
old='Figure~\\ref{fig:tdi_log_amplitudes} follows the waveform absolute values through three dominant-mode damping times after the joint clean-envelope peak. This interval resolves the principal decay and zero crossings; Fig.~\\ref{fig:examples} shows the full observations. All A/E reconstruction errors use the complete 528-s records. '
assert old in s;s=s.replace(old,'')
start=s.index('Figure~\\ref{fig:conditional_strain} connects')
end=s.index('\\begin{figure*}',start)
s=s[:start]+r'''Figure~\ref{fig:conditional_frequency} compares the detector-channel and incident-strain reconstructions for sources with dominant-mode frequencies of 7.3 and 38.7 mHz. Their damping times are 135.6 and 26.7 s. A common observer-frame time axis displays the difference in oscillation period and decay rate. Both examples have $\rho\ge10$, direct mismatch below $5\times10^{-3}$ and gap-local relative error below 15\%. The full-record $A/E$ mismatches are $2.60\times10^{-4}$ and $2.63\times10^{-3}$, respectively. Over the displayed 240 s, the joint-polarization relative errors $\|\widehat s-s\|_2/\|s\|_2$ are 1.24\% and 5.35\%.

The strain curves include the projected noise realization $s+Pn$, evaluated with the same instrument noise and response-basis projection as in Eq.~\eqref{eq:conditional_strain}. Its smooth, correlated fluctuations follow from the finite QNM basis. The source parameters remain fixed at their injected values throughout this comparison; Table~\ref{tab:results} evaluates the direct $A/E$ reconstruction across the test population.

'''+s[end:]
start=s.index(r'\caption{Conditional strain reconstruction at lower')
end=s.index(r'\label{fig:conditional_frequency}',start)
s=s[:start]+r'''\caption{Detector-channel and conditional incident-strain reconstructions at dominant-mode frequencies of 7.3 mHz (left) and 38.7 mHz (right). The upper rows overlay the noisy observation $Gh+n$ (gray), clean signal $h$ (blue), gapped signal $Gh$ (green dash-dotted) and direct DGF reconstruction $\widehat h$ (orange dashed). Their axes give fractional frequency in units of $10^{-18}$ over the full 528-s records; shading marks the gaps. The lower rows show the injected polarizations (blue), conditional reconstruction $P\widehat h$ (orange dashed) and strain plus projected noise $s+Pn$ (gray). Strain units are $10^{-18}$ on the left and $10^{-19}$ on the right. Both source panels use observer-frame time from the mode-amplitude reference epoch. Gray fill spans the injected strain and one projected noise realization. Curves use the native 0.5-s samples and the original noise amplitudes.}'''+s[end:]
s=s.replace(' Source seeds are 2042203874, 2042204884 and 2042204076. Appendix~\\ref{app:model_analysis} gives the example criteria.','')
s=s.replace('for the $A$ channel of source 2042203874, also shown in Fig.~\\ref{fig:examples}(a)','for the $A$ channel of the first example in Fig.~\\ref{fig:examples}')
# Replace implementation identifiers with a concise availability statement.
start=s.index('Source seeds are consecutive integers')
end=s.index('The three examples in Fig.',start)
s=s[:start]+r'''The implementation uses PyCBC, LALSuite, \texttt{fastlisaresponse}, LISA Analysis Tools, \texttt{ssqueezepy} and \texttt{pytorch\_wavelets}. Code is maintained at \url{https://github.com/ljq2088/RingdownGapFilling}. Trained models, calibration arrays and evaluation records are available from the authors on request.

'''+s[end:]
start=s.index('The three examples in Fig.',s.index(r'\appendix'))
end=s.index('\n\n',start)
s=s[:start]+r'''The waveform examples in Fig.~\ref{fig:examples} have resolved oscillations, $\rho\ge10$, gap-local relative errors below 7\% and channel peak-phase errors below $2.9^\circ$. The incident-strain comparison in Sec.~\ref{subsec:incident_strain} illustrates recovery at two separated frequencies. Population statistics include every test source.'''+s[end:]
s=s.replace('The encoder and waveform decoder are initialized from DGF weights identified in the accompanying manifest.','The encoder and waveform decoder are initialized from a DGF model pretrained on auxiliary ringdown simulations.')
s=s.replace('The physical training, validation and test populations use the disjoint seed ranges given above.','The physical training, validation and test sets contain independently generated source populations.')
s=s.replace(' Their configurations and weights are identified in the manifest.','')
s=s.replace('The conditioned coefficient head starts from weights \\texttt{a1e9959a7a95}. It is fitted with the waveform estimator fixed and then jointly with the decoder and refinement modules,','The conditioned coefficient head is first trained with the waveform estimator fixed and then jointly with the decoder and refinement modules,')
s=s.replace(' Its configuration and checkpoint are recorded in the manifest.','')
background='Black hole ringdown carries information about the remnant mass and spin and provides a basis for tests of general relativity. Space-based detectors offer access to the ringdowns of massive black hole binaries in the millihertz band. '
s=s.replace('\\begin{abstract}\nData gaps','\\begin{abstract}\n'+background+'Data gaps',1)
# Keep paragraphs compact after removing complete figure blocks.
s=re.sub(r'\n{4,}','\n\n\n',s)
assert len(re.findall(r'\\begin\{figure\*\}',s))==8
assert not re.search(r'\bseeds?\b|SHA-256|204220\d+|a1e9959a7a95|96a8ea344c45|fig:conditional_strain|fig:tdi_log_amplitudes',s,re.I)
assert background in s
assert re.findall(r'\\begin\{table\*?\}.*?\\end\{table\*?\}',original,re.S)==re.findall(r'\\begin\{table\*?\}.*?\\end\{table\*?\}',s,re.S)
f.write_text(s,encoding='utf-8',newline='\n')
f=P/'source/reproducibility_manifest.json';v=json.loads(f.read_text(encoding='utf-8'));v['editorial_streamlining']={'removed_from_article':['tdi_log_amplitudes_20261015.pdf','conditional_source_strain.pdf'],'retained_waveform_views':['latest_examples.pdf','conditional_frequency_contrast.pdf'],'reproducibility_identifiers':'Retained in this manifest and project records; omitted from manuscript prose and captions.','figures':8,'abstract':'Added ringdown motivation and millihertz relevance.','statistics':'unchanged'};f.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8',newline='\n')
script=(P.parent/'ringdown_maintext_20261015/stage_compile.py').read_text(encoding='utf-8').replace("'ringdown_maintext_20261015'","'ringdown_streamlined_20261015'");(P/'stage_compile.py').write_text(script,encoding='utf-8')
print(re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',s,re.S).group(1))
