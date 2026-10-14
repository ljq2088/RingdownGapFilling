from pathlib import Path
import zipfile,json,re,difflib
here=Path(__file__).resolve().parent
with zipfile.ZipFile(here/'overleaf_before.zip') as z:old=z.read('DGF.tex').decode('utf-8')
tex=old
start=tex.index(r'Figure~\ref{fig:complementary_examples} gives sources');end=tex.index(r'\end{figure*}',start)+len(r'\end{figure*}')
tex=tex[:start]+tex[end:]
tex=tex.replace(r' Appendix~\ref{app:complementary} gives waveform comparisons at the median, 90th-percentile and largest-mismatch ranks of the test population.','')
start=tex.index(r'Figure~\ref{fig:examples} compares');end=tex.index('\n',start)
paragraph=r'Figure~\ref{fig:examples} compares the clean and reconstructed waveforms across 36- and 45-s gaps, with noise-weighted mismatches $5.59\times10^{-4}$ and $6.07\times10^{-4}$. The gaps contain 42.7\% and 56.8\% of the respective joint signal squared norms. Their gap-local relative errors are 4.6\% and 3.7\%, and both channels retain the decaying oscillations through the missing intervals. The windows extend for approximately 4.9 dominant-mode damping times after the response peak. Logarithmic-amplitude views are given in Appendix~\ref{app:complementary}.'
tex=tex[:start]+paragraph+tex[end:]
label=tex.index(r'\label{fig:examples}');start=tex.rfind(r'\caption{',0,label)
caption=r'\caption{Waveform reconstruction across 36- and 45-s gaps in the $A$ (left) and $E$ (right) channels. Each panel superposes the noisy observation $d=Gh+n$ (gray), clean signal $h$, gapped clean signal $Gh$ and DGF reconstruction $\widehat h$ over the full 528-s record; insets resolve the gap. Shading marks signal suppression, with noise retained. All four curves use the same clean-signal amplitude normalization per source and connect the original 0.5-s samples. Source seeds, from top to bottom, are 2038202231 and 2038201952. Full-population statistics use all 6,000 test sources.}'
tex=tex[:start]+caption+tex[label:]
tex=tex.replace('source 2038205946, also shown','source 2038202231, also shown')
tex=tex.replace(r'approximately 45\%--57\%',r'approximately 43\%--57\%')
tex=tex.replace('The three sources in Fig.', 'The two sources in Fig.')
tex=tex.replace(r'Their global and gap-local relative errors are below 9\%, and their per-channel peak phase errors are below $1.5^\circ$.',r'Their global and gap-local relative errors are below 6\%, and their per-channel peak phase errors are below $2.5^\circ$.')
assert all(s not in tex for s in ['complementary_examples','latest_diagnostic_examples','upper strip','2038205946','2038202314','Selected successful'])
(here/'source/DGF.tex').write_text(tex,encoding='utf-8',newline='\n')
(here/'source/latest_diagnostic_examples.pdf').unlink(missing_ok=True)
(here/'manuscript.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),tex.splitlines(True),fromfile='before/DGF.tex',tofile='after/DGF.tex')))
print('Removed old figure 8; restored observation overlay; updated two-source examples.')
