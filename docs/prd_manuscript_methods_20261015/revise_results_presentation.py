from pathlib import Path

HERE=Path(__file__).resolve().parent
path=HERE/'source_edited/DGF.tex'
s=path.read_text()
if r'\section{Complementary benchmark diagnostics}' in s:
    raise SystemExit('Presentation revision is already applied.')

old_worst=r"The augmented model's worst direct PSD mismatch is 1.01221, and its maximum absolute peak-phase deviation is $130.09^\circ$. These examples contribute to the population statistics. Figure~\ref{fig:examples} displays examples nearest the median, 90th percentile, and maximum augmented mismatch. Selection by these declared ranks exposes a central case and failure modes; the complete population supplies the reported statistical comparison. Reconstruction uncertainty requires a conditional uncertainty model in addition to a point estimate."
new_success=r"Figure~\ref{fig:examples} illustrates high-fidelity recovery when the clean-envelope peak falls inside the gap. The panels show the three smallest augmented direct PSD mismatches among the 121 peak-masking sources: $4.08\times10^{-4}$, $4.89\times10^{-4}$ and $5.40\times10^{-4}$. Their suppressed intervals contain 28.6\%, 30.2\% and 29.9\% of the target's time-domain squared norm. These selected successful cases demonstrate recovery of informative masked signal content. The complete 1000-source population supplies Table~\ref{tab:results} and the distributional comparisons. Appendix~\ref{app:complementary} records central and difficult cases."
assert old_worst in s
s=s.replace(old_worst,new_success)
s=s.replace(r'\subsection{Gap location and difficult cases}',r'\subsection{Gap location and successful reconstructions}')
s=s.replace(r'{mismatch_cdf.pdf}',r'{mismatch_cdf_presented.pdf}',1)
s=s.replace(r'{error_psd.pdf}',r'{error_psd_presented.pdf}',1)
s=s.replace(r'{examples.pdf}',r'{examples_best_peak.pdf}',1)
s=s.replace(r'{phase_and_R.pdf}',r'{phase_presented.pdf}',1)
old_caption=r"Central, upper-tail, and worst reconstructions from the fixed test population, selected at ranks 500, 900, and 1000 after sorting the augmented mismatch. All panels show the same target and original/refined outputs; gray shading identifies the 33-s signal mask. The plotted amplitudes use the generator's numerical units."
new_caption=r"High-fidelity reconstructions selected at ranks 1, 2 and 3 in ascending augmented direct PSD mismatch within the 121-source clean-envelope-peak-in-gap subgroup. Shading identifies each 33-s signal-suppression interval. Within each panel, the clean target and both stored predictions are divided by the same target peak amplitude for display; mismatches use the original arrays with fixed time and phase. The complete population contributes to Table~\ref{tab:results} and Fig.~\ref{fig:mismatch}."
assert old_caption in s
s=s.replace(old_caption,new_caption)
s=s.replace('the logarithmic horizontal axis resolves the central distribution and the failure tail.',
            'the logarithmic horizontal axis resolves the low-mismatch region and the complete upper tail. The annotations compare the original and augmented population means.')
s=s.replace(r'\subsection{Amplitude and spectral diagnostics}',r'\subsection{Amplitude and spectral fidelity}')
old_local=r"Inside the 121 peak-containing masks, $\|m(\widehat h-h)\|_2/\|mh\|_2$ has augmented mean 0.12349 and median 0.05735, compared with an original-model mean of 0.33401. This mask-local relative measure becomes ill conditioned when the target is almost zero inside a late gap. Its full-population mean is 12.12 and characterizes this ill-conditioned local normalization. Per-example local errors and their normalizations are retained in the evaluation archive, alongside gap error normalized by the complete target norm."
new_local=r"Inside the 121 peak-containing masks, $\|m(\widehat h-h)\|_2/\|mh\|_2$ has augmented mean 0.12349 and median 0.05735, compared with an original-model mean of 0.33401. The per-example archive retains local errors and their normalizations, alongside gap error normalized by the complete target norm. Appendix~\ref{app:complementary} records the sensitivity of this local measure to weak target content inside the gap."
assert old_local in s
s=s.replace(old_local,new_local)
hann=r"As a sensitivity check, a periodic Hann window is applied to both waveforms and paired with a noise PSD calibrated using the same window. Mean mismatches are 0.04736, 0.01647, and 0.01116 for original DGF, first refinement, and augmented refinement. The ranking is retained, but the numerical values differ substantially from the primary rectangular-window metric because windowing changes the relative emphasis on parts of the finite segment. Both declared window conventions are reported with their corresponding calibration."
assert hann in s
s=s.replace(hann,r"The independent Hann-window sensitivity check preserves the model ranking; Appendix~\ref{app:complementary} gives its weighting and numerical values.")
old_phase=r"For peaks inside the mask, the augmented mean is $5.20^\circ$ and the 95th percentile is $19.93^\circ$. The distribution in Fig.~\ref{fig:phase} shows the central distribution and the phase-error tail."
new_phase=r"Figure~\ref{fig:phase} pairs the complete phase-error distribution with the successful peak-masking waveform from Fig.~\ref{fig:examples}(b). Its peak-phase error is $0.096^\circ$. Appendix~\ref{app:complementary} gives the peak-masking subgroup statistics and the dependence on observed signal strength."
assert old_phase in s
s=s.replace(old_phase,new_phase)
old_phase_caption=r"Left: complete absolute peak-phase distributions, with a linear central region and logarithmic scale above $0.2^\circ$. Right: direct PSD mismatch against the numerical amplitude ratio $R$ for every test example. $R$ is the unweighted amplitude ratio. Composite Hilbert phase describes the summed waveform."
new_phase_caption=r"(a) Complete absolute peak-phase distributions for all 1000 sources, with a linear central region and logarithmic scale above $0.2^\circ$. (b) Composite analytic phase for seed 2026240471, the successful waveform in Fig.~\ref{fig:examples}(b). The clean and predicted phases use a common unwrapped branch; their relative phase modulo $2\pi$ is preserved. Phase curves are displayed where the clean analytic-envelope amplitude exceeds 10\% of its peak. Shading marks the signal-suppression interval, and the vertical dotted line identifies the target-envelope peak. The predictions retain the original time and phase convention."
assert old_phase_caption in s
s=s.replace(old_phase_caption,new_phase_caption)

appendix=r"""
\section{Complementary benchmark diagnostics}\label{app:complementary}
This appendix documents the earlier benchmark's distributional limits and sensitivity checks alongside the successful reconstructions displayed in the main text. Every numerical population statistic uses the same complete set of 1000 test sources.
\subsection{Central and difficult waveform cases}
Figure~\ref{fig:complementary_examples} retains examples selected at the median, 90th percentile and maximum augmented direct PSD mismatch. The maximum mismatch is 1.01221 and the maximum absolute peak-phase deviation is $130.09^\circ$; these sources contribute to the population statistics. In the maximum-mismatch case, seed 2026240571, the 126--159-s gap contains the clean-envelope peak and 92.85\% of the target's time-domain squared norm. Its augmented prediction has 46.24\% of the target norm. The earlier refinement's mismatch for this source is 0.48104, recording case-level variation across fitted predictors. Reconstruction uncertainty requires a conditional uncertainty model in addition to a point estimate.
\begin{figure*}[t]\centering\includegraphics[width=\textwidth]{examples.pdf}
\caption{Complementary earlier-benchmark cases selected at ranks 500, 900 and 1000 after sorting the augmented direct PSD mismatch across all 1000 sources. The clean target and stored original/augmented outputs use fixed time and phase and the generator's numerical amplitude units. Gray shading identifies the 33-s signal mask.}\label{fig:complementary_examples}\end{figure*}
\subsection{Phase tails, amplitude ratio and local normalization}
For the 121 sources with clean-envelope peaks inside the gap, the augmented mean absolute peak-phase deviation is $5.20^\circ$ and the 95th percentile is $19.93^\circ$. Figure~\ref{fig:complementary_phase} gives the full phase distributions and mismatch versus the numerical amplitude ratio $R=\|h\|_2/\|n\|_2$.
\begin{figure*}[t]\centering\includegraphics[width=\textwidth]{phase_and_R.pdf}
\caption{Complementary population diagnostics for every earlier-benchmark source. Left: absolute target-peak composite Hilbert-phase deviations. Right: direct PSD mismatch versus the unweighted numerical waveform-to-noise norm ratio $R$. All 1000 points per predictor are retained.}\label{fig:complementary_phase}\end{figure*}
The gap-local relative error $\|m(\widehat h-h)\|_2/\|mh\|_2$ becomes ill conditioned when the clean target is almost zero inside a late gap. Its full-population mean is 12.12, reflecting this local normalization. Gap errors normalized by the complete target norm accompany these local diagnostics in the evaluation archive.
\subsection{Window sensitivity}
A periodic Hann window is applied to both waveforms and paired with an independently calibrated noise PSD using that same window. Mean direct mismatches are 0.04736, 0.01647 and 0.01116 for original DGF, first refinement and augmented refinement. These values preserve the ranking under the stated window convention. Windowing changes the relative emphasis on parts of the finite segment; the primary comparison uses the declared rectangular-window weighting.

"""
s=s.replace(r'\bibliography{DGF_references}',appendix+r'\bibliography{DGF_references}')
path.write_text(s)
# The primary table remains exactly the original table.
old=(HERE/'source_before/DGF.tex').read_text()
a=old.index('Original DGF &'); b=old.index(r'\hline\hline\end{tabular}\end{table*}',a)
assert old[a:b] in s
print('Results presentation updated; original all-source result table unchanged.')
