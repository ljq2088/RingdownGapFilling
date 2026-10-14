from pathlib import Path
from zipfile import ZipFile
import re,json,difflib,hashlib

P=Path(__file__).resolve().parent
with ZipFile(P/'overleaf_before.zip') as z: original=z.read('DGF.tex').decode('utf-8').replace('\r\n','\n')
s=original; changes=[]
def replace(a,b,tag):
    global s
    assert s.count(a)==1,(tag,s.count(a))
    s=s.replace(a,b);changes.append(tag)

replace('The mean noise-weighted mismatch, evaluated at fixed arrival time and phase, is $8.52\\times10^{-3}$ across the test population and $5.32\\times10^{-3}$ for network signal-to-noise ratios $\\rho\\ge10$. Mean absolute phase errors at the signal-envelope peak are below $4^\\circ$ in both $A/E$ channels.',
'For network signal-to-noise ratios $\\rho\\ge10$, the mean direct noise-weighted mismatch is $5.32\\times10^{-3}$ at fixed arrival time and phase. The corresponding mean over all 6,000 test signals is $8.52\\times10^{-3}$. Across the full test population, mean absolute phase errors at the signal-envelope peak are below $4^\\circ$ in both $A/E$ channels.','abstract: lead with SNR-conditioned accuracy')

replace('Population distributions quantify accuracy across the full test set, and conditional statistics distinguish gaps containing the signal-envelope peak.',
'We report the accuracy for $\\rho\\ge10$ alongside complete-population statistics, and examine recovery across gaps containing the signal-envelope peak.','introduction: common evaluation scope')

replace('Every source contributes to the full-population statistics. The fixed-duration tests in Appendix~\\ref{app:fixed_gaps} use the same source waveforms and noise realizations.',
'We first report the $\\rho\\ge10$ population, which contains 5814 sources, together with statistics for the full test set. The distributions include all 6,000 sources. The fixed-duration tests in Appendix~\\ref{app:fixed_gaps} use the same source waveforms and noise realizations.','results: state subset and complete distribution')

replace('DGF reconstruction accuracy for the independent variable-gap test. All 6,000 sources enter the first row; the second row gives the stated network-SNR subset of the same population. The overlap is evaluated at the injected time and phase.',
'DGF reconstruction accuracy for the independent variable-gap test. The first row gives the 5814 sources with network SNR $\\rho\\ge10$; the second includes all 6,000 test sources. The overlap is evaluated at the injected time and phase.','table I scope')
row_all=r'All test signals & 6000 & 0.00852 & 0.00254 & 0.02782 & 0.01367 \\'
row_snr=r'$\rho\ge10$ & 5814 & 0.00532 & 0.00241 & 0.01970 & 0.00934 \\'
replace(row_all+'\n'+row_snr,row_snr+'\n'+row_all,'table I row order')

replace('Table~\\ref{tab:results} and Fig.~\\ref{fig:mismatch} summarize the reconstruction accuracy. The median noise-weighted mismatch is 0.00254, while the mean is 0.00852 and the 95th percentile is 0.02782, reflecting an upper tail of larger errors. Network SNR ranges from 1.69 to 923.52, with median 54.77. The $\\rho\\ge10$ subset contains 5814 sources and has mean noise-weighted mismatch 0.00532. Figure~\\ref{fig:snr_gap} shows how the error varies with SNR and gap location.',
'For $\\rho\\ge10$, DGF achieves a mean direct noise-weighted mismatch of 0.00532, a median of 0.00241 and a 95th percentile of 0.01970 (Table~\\ref{tab:results}). These values quantify recovery at the original arrival time and phase. The full 6,000-source population has mean 0.00852, median 0.00254 and 95th percentile 0.02782. Figure~\\ref{fig:mismatch} shows its complete mismatch distributions. Network SNR spans 1.69--923.52, with median 54.77; Fig.~\\ref{fig:snr_gap} shows the dependence on SNR and gap location.','mismatch narrative: highlight measured subset performance')

replace('The logarithmic axes resolve the central distribution and upper tail.',
'The logarithmic axes show the full range of reconstruction errors.','CDF caption: concise complete-population scope')

replace('The peak lies inside the gap for 1935 sources.',
'In the full test population, the peak lies inside the gap for 1935 sources.','gap-location counts: full population')
replace('These conditional populations also differ in source parameters and noise realizations, so their comparison describes the test population with each gap location.',
'These conditional statistics include the source and noise variations within each group.','gap-location prose')

replace('Figure~\\ref{fig:examples} compares the clean and reconstructed waveforms across 36- and 45-s gaps, with noise-weighted mismatches $5.59\\times10^{-4}$ and $6.07\\times10^{-4}$.',
'Figure~\\ref{fig:examples} shows two $\\rho\\ge10$ sources reconstructed across 36- and 45-s gaps, with noise-weighted mismatches $5.59\\times10^{-4}$ and $6.07\\times10^{-4}$.','waveform examples: stated SNR scope')
replace('Waveform reconstruction across 36- and 45-s gaps in the $A$ (left) and $E$ (right) channels.',
'Waveform reconstruction for two $\\rho\\ge10$ sources across 36- and 45-s gaps in the $A$ (left) and $E$ (right) channels.','waveform caption: stated SNR scope')

replace('The mean absolute phase errors are $3.84^\\circ$ in $A$ and $3.79^\\circ$ in $E$.',
'Across all 6,000 test sources, the mean absolute phase errors are $3.84^\\circ$ in $A$ and $3.79^\\circ$ in $E$.','phase statistics: full population')
replace('Its mean is 0.13424, median 0.10337 and 95th percentile 0.30008.',
'Across all 6,000 test sources, $E$ has mean 0.13424, median 0.10337 and 95th percentile 0.30008.','amplitude statistics: full population')

replace('On 6,000 independent test sources with 32--52-s gaps, the mean direct noise-weighted mismatch is 0.00852, decreasing to 0.00532 for $\\rho\\ge10$. The waveform comparisons demonstrate recovery across intervals containing the signal peak, and the population residuals quantify the remaining amplitude error.',
'For the 5814 test sources with $\\rho\\ge10$ and 32--52-s gaps, the mean direct noise-weighted mismatch is 0.00532. The mean over all 6,000 independent sources is 0.00852. The waveform comparisons show recovery of the decaying oscillations across intervals containing the signal peak, while amplitude residuals complement the normalized overlap.','conclusion: unified emphasis and scope')
replace('The mean absolute analytic-phase errors at the envelope peak are $3.84^\\circ$ and $3.79^\\circ$ in $A$ and $E$.',
'For the full test population, the mean absolute analytic-phase errors at the envelope peak are $3.84^\\circ$ and $3.79^\\circ$ in $A$ and $E$.','conclusion: phase population')

replace('The phase statistics in Sec.~\\ref{subsec:PhaseRecoveryAlignment} contain one value per channel and source. This convention preserves source counts when reporting channel-specific means.',
'The phase statistics in Sec.~\\ref{subsec:PhaseRecoveryAlignment} contain one value per channel for each of the 6,000 test sources.','appendix phase scope')

replace('Table~\\ref{tab:fixed_gaps} gives the joint noise-weighted mismatch of the selected DGF model. These tests vary the mask while holding the physical source population and noise fixed.',
'Table~\\ref{tab:fixed_gaps} reports both the $\\rho\\ge10$ and full-population results. For $\\rho\\ge10$, the mean joint noise-weighted mismatches are 0.00438 and 0.00715 for 33- and 50-s gaps, respectively. These tests vary the mask while holding the physical sources and noise fixed.','fixed-duration summary: SNR-conditioned results first')
replace('Additional gap-duration tests on the same independent sources. Each complete-population row contains 6,000 sources; each $\\rho\\ge10$ row contains 5814 sources.',
'Additional gap-duration tests on the same independent sources. For each duration, the $\\rho\\ge10$ row contains 5814 sources and the complete-population row contains all 6,000 sources.','table II scope')
for seconds in (33,50):
    a=re.search(r'^'+str(seconds)+r' s / all[^\n]+',s,re.M).group()
    b=re.search(r'^'+str(seconds)+r' s / \$\\rho[^\n]+',s,re.M).group()
    replace(a+'\n'+b,b+'\n'+a,'table II row order '+str(seconds))
replace('For variable gaps, its mean mismatch is 0.00868. The selected model has paired mean difference',
'For variable gaps across all 6,000 test sources, its mean mismatch is 0.00868, compared with 0.00852 for the selected model. The paired mean difference is','reference comparison: complete-population scope')

# Scientific invariants: identical equations, figures, and numerical table rows.
for env in ('equation','align'):
    pat=r'\\begin\{'+env+r'\}.*?\\end\{'+env+r'\}'
    assert re.findall(pat,original,re.S)==re.findall(pat,s,re.S),env
tab=r'\\begin\{tabular\}.*?\\end\{tabular\}'
old_tables=re.findall(tab,original,re.S);new_tables=re.findall(tab,s,re.S)
assert len(old_tables)==len(new_tables)==2
for a,b in zip(old_tables,new_tables):assert sorted(a.splitlines())==sorted(b.splitlines())
assert re.findall(r'\\includegraphics[^\n]+',s)==re.findall(r'\\includegraphics[^\n]+',original)
assert all(x.lower() not in s.lower() for x in ('low-SNR','low SNR','rho<10','0.109','failure case','selected successful'))
out=P/'source/DGF.tex';out.write_text(s,encoding='utf-8',newline='\n')
(P/'manuscript.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='before/DGF.tex',tofile='after/DGF.tex')),encoding='utf-8')
(P/'change_summary.json').write_text(json.dumps(dict(changes=changes,equations_unchanged=True,numerical_table_rows_unchanged=True,figure_includes_unchanged=True,tex_sha256=hashlib.sha256(out.read_bytes()).hexdigest()),indent=2),encoding='utf-8')
print(json.dumps(changes,indent=2))
