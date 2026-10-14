"""Bind the complete paper to the selected model and latest independent test."""
from pathlib import Path
import json, re

HERE=Path(__file__).resolve().parent
p=HERE/'source_edited/DGF.tex'
base=p.read_text(encoding='utf-8')
(HERE/'DGF_before_unification.tex').write_text(base,encoding='utf-8')
s=json.loads((HERE/'latest_test_summary.json').read_text())
text=base
def replace_block(start,end,value):
    global text
    a=text.index(start);b=text.index(end,a)
    text=text[:a]+value.strip()+'\n\n'+text[b:]
def five(x):return f'{x:.5f}'
def sci(x):
    a,b=f'{x:.2e}'.split('e')
    return a+r'\times10^{'+str(int(b))+'}'
primary=s['primary']; peak=s['peak_in_gap']; outside=s['peak_outside_gap']; strong=s['snr_ge10']

replace_block(r'\begin{abstract}',r'\maketitle',rf'''
\begin{{abstract}}
Data gaps can obscure the rapidly decaying oscillations of a black hole ringdown. We present DenoiseGapFiller (DGF), a deep-learning framework for joint denoising and waveform reconstruction. Its dual-branch encoder combines time--frequency and multiscale temporal representations; a waveform decoder and mask-aware residual blocks estimate the complete signal. Simulated data are generated with standard LISA response and noise models. On independent test data with gaps spanning approximately 6\%--10\% of the analysis window, DGF achieves a mean noise-weighted mismatch of ${sci(primary['psd']['mean'])}$. For signals with network signal-to-noise ratio $\rho\ge10$, the mean mismatch is ${sci(strong['psd']['mean'])}$. Mean absolute phase errors at the signal-envelope peak are below $4^\circ$ in both channels. DGF recovers the oscillatory structure across gaps and provides waveform estimates for subsequent ringdown analysis.
\end{{abstract}}
''')

old='The quantitative results reported here use the controlled waveform and noise prescription in Appendix~\\ref{app:sequence_modeling}; the physical LISA source and channel conventions are specified in Sec.~\\ref{sec:method}.'
new='All reconstruction results use one selected DGF model and the independent LISA test population defined in Sec.~\\ref{sec:method}. The source and channel conventions are common to training, validation and testing.'
assert old in text;text=text.replace(old,new)
text=text.replace('The appendixes provide implementation details, the controlled waveform prescription, complementary diagnostics and training comparisons.', 'The appendixes provide implementation details, additional waveform diagnostics and tests with fixed gap durations.')
a=text.index(r'\section{Methodology}\label{sec:method}');b=text.index(r'\subsection{Physical ringdown signals',a)
text=text[:a]+r'''\section{Methodology}\label{sec:method}
We generate simulated ringdown observations using standard LISA response and noise models. DGF operates on the $A/E$ TDI channels and estimates their complete clean waveforms from the observations and known gap mask. The signal, response, noise calibration and preprocessing described below are shared by the training, validation and independent test sets.
'''+ '\n'+text[b:]
text=text.replace('Four fundamental modes, $(\\ell,m,n)=(2,2,0),(2,1,0),(3,3,0),(4,4,0)$, enter', 'The fundamental modes $(\\ell,m,n)=(2,2,0),(2,1,0),(3,3,0),(4,4,0)$ enter')
text=text.replace(r'\subsection{Analysis windows and variable signal-suppression intervals}',r'\subsection{Analysis windows and data gaps}')
text=text.replace(r'\mathcal M_{AE}',r'\mathcal M')
text=text.replace('Appendix~\\ref{app:training_checks} describes feature selection and supplementary training comparisons.', 'Appendix~\\ref{app:fixed_gaps} reports supplementary evaluation on the same independent test sources.')
text=text.replace('The test protocol reserves 6,000 independent source seeds and generates their observations after model selection.', 'The independent test set contains 6,000 sources generated after model selection.')
text=text.replace('composite analytic phase', 'per-channel analytic phase')
text=text.replace('The LISA configuration uses a two-stage procedure.', 'The model is initialized from a pretrained DGF reconstruction model, with a zero-initialized output layer in the noise-conditioned coefficient head. Training proceeds in two stages.')
text=text.replace('The implementation is available at \\url{https://github.com/ljq2088/RingdownGapFilling}; configurations and checkpoint identifiers are included with the evaluation record.', 'Configurations, source seeds and frozen checkpoint identifiers are included in the evaluation record. The project repository is \\url{https://github.com/ljq2088/RingdownGapFilling}.')

rows=[]
for label,n,v,t in [('All test signals',6000,primary['psd'],primary['time']),
                    (r'$\rho\ge10$',strong['n'],strong['psd'],strong['time'])]:
    rows.append(f"{label} & {n} & {five(v['mean'])} & {five(v['median'])} & {five(v['p95'])} & {five(t['mean'])} "+r'\\')
cases=s['selected_cases']; casevals=', '.join('$'+sci(c['mismatch'])+'$' for c in cases)
energies=', '.join(f"{100*c['masked_energy_fraction']:.1f}\\%" for c in cases)
results=rf'''
\section{{Reconstruction results}}\label{{sec:results}}
All results in this section and the appendixes use the same selected DGF model and 6,000 independent LISA test sources. Training and validation sources are disjoint from this test population. Each record contains 1056 samples at 2 Hz, corresponding to 528 s. The primary test assigns one gap of 64--104 samples, sampled uniformly from the even lengths in this range, giving durations of 32--52 s. The model is evaluated once after validation selection, and every source contributes to the reported population statistics.

\subsection{{Waveform mismatch}}\label{{subsec:Mismatch}}
We evaluate the noise-weighted mismatch in Eq.~\eqref{{eq:joint_mismatch}} jointly over the $A/E$ channels at the injected time and phase. The corresponding time-domain mismatch is
\begin{{equation}}
\mathcal M_t=1-\frac{{\sum_{{c,j}}h_{{c,j}}\widehat h_{{c,j}}}}{{\sqrt{{\sum_{{c,j}}h_{{c,j}}^2\sum_{{c,j}}\widehat h_{{c,j}}^2}}}}.
\end{{equation}}
These normalized overlaps measure waveform agreement. Population means assign equal weight to each source. We report amplitude-sensitive residuals separately in Sec.~\ref{{subsec:MeanSquaredError}}.

\begin{{table*}}[t]\centering
\caption{{DGF reconstruction accuracy for the independent variable-gap test. All 6,000 sources enter the first row; the second row gives the stated network-SNR subset of the same population. The overlap is evaluated at the injected time and phase.}}\label{{tab:results}}
\begin{{tabular}}{{lrrrrr}}\hline\hline
Population & $n$ & Mean $\mathcal M$ & Median $\mathcal M$ & 95th percentile $\mathcal M$ & Mean $\mathcal M_t$\\\hline
{chr(10).join(rows)}
\hline\hline\end{{tabular}}\end{{table*}}

The mean noise-weighted mismatch is {five(primary['psd']['mean'])}, with median {five(primary['psd']['median'])} and 95th percentile {five(primary['psd']['p95'])}. The mean time-domain mismatch is {five(primary['time']['mean'])}. Figure~\ref{{fig:mismatch}} gives the complete distributions. Network SNR, defined by Eq.~\eqref{{eq:joint_mismatch}}, ranges from {s['snr']['min']:.2f} to {s['snr']['max']:.2f}, with median {s['snr']['median']:.2f}. The $\rho\ge10$ subset contains {strong['n']} sources and has mean noise-weighted mismatch {five(strong['psd']['mean'])}. Figure~\ref{{fig:snr_gap}} resolves the dependence on SNR and gap location.
\begin{{figure*}}[t]\centering\includegraphics[width=\textwidth]{{latest_mismatch.pdf}}
\caption{{Empirical distributions of time-domain and noise-weighted mismatch for all 6,000 independent test signals. The logarithmic axes resolve the central distribution and upper tail. Both metrics use the original signal time and phase.}}\label{{fig:mismatch}}\end{{figure*}}

\subsection{{Gap location and waveform recovery}}
We characterize gap location using the peak of the joint analytic envelope $[\sum_c|h_c+i\mathcal H[h_c]|^2]^{{1/2}}$, where $\mathcal H$ denotes the Hilbert transform. The peak lies inside the gap for {peak['n']} sources. Their mean noise-weighted mismatch is {five(peak['psd']['mean'])}, with median {five(peak['psd']['median'])}. For the remaining {outside['n']} sources, the mean is {five(outside['psd']['mean'])}. These conditional distributions quantify reconstruction when the strongest part of the signal is suppressed, while retaining the source and noise variation within each group.

Figure~\ref{{fig:examples}} illustrates the three smallest noise-weighted mismatches within the peak-inside-gap subgroup: {casevals}. The gaps contain {energies} of the respective joint signal squared norms. The reconstructed $A/E$ waveforms follow the missing signal through these informative intervals. These selected cases illustrate successful recovery; the tables and distribution plots include the complete population. Central and upper-tail cases are shown in Appendix~\ref{{app:complementary}}.
\begin{{figure*}}[t]\centering\includegraphics[width=\textwidth]{{latest_examples.pdf}}
\caption{{Successful reconstructions in the $A$ (left) and $E$ (right) channels. The three sources have the smallest joint noise-weighted mismatches among the {peak['n']} test sources whose joint envelope peaks lie inside the gap. Shading marks each source's gap. For display, both channels and their predictions are divided by the same maximum clean-signal amplitude within each source. Metrics use the original physical arrays.}}\label{{fig:examples}}\end{{figure*}}

\subsection{{Amplitude and spectral fidelity}}\label{{subsec:MeanSquaredError}}\label{{subsec:AmplitudeDensityFrequencySpectrum}}
Normalized overlap is invariant under positive amplitude rescaling. We therefore evaluate the relative waveform error
\begin{{equation}}
E=\frac{{\left[\sum_{{c,j}}(\widehat h_{{c,j}}-h_{{c,j}})^2\right]^{{1/2}}}}{{\left[\sum_{{c,j}}h_{{c,j}}^2\right]^{{1/2}}}}.
\end{{equation}}
Its mean is {primary['relative_error']['mean']:.5f}, median {primary['relative_error']['median']:.5f} and 95th percentile {primary['relative_error']['p95']:.5f}. The mean signed norm bias, $\|\widehat h\|_2/\|h\|_2-1$, is {primary['norm_bias']['mean']:.5f}. Together with mismatch, these measurements characterize waveform shape and overall amplitude fidelity.

For the {peak['n']} sources with envelope peaks inside the gap, the gap-local relative error $\|m(\widehat h-h)\|_2/\|mh\|_2$ has mean {peak['gap_relative_error']['mean']:.5f} and median {peak['gap_relative_error']['median']:.5f}. The denominator measures the clean signal contained within the interval. The full-waveform error $E$ remains the common amplitude-sensitive measure for all sources.

Figure~\ref{{fig:psd}} compares the mean one-sided periodograms of the reconstruction residual and simulated instrument noise in each channel. Both spectra average all 6,000 test sources and are expressed in fractional-frequency units. They show the frequency dependence of the residual after joint denoising and reconstruction. Mode-resolved frequency and damping-time estimation requires fitting the individual ringdown components.
\begin{{figure*}}[t]\centering\includegraphics[width=\textwidth]{{latest_residual_psd.pdf}}
\caption{{Mean residual and noise power spectral densities in the $A/E$ channels over all 6,000 test sources. The reconstruction residual is $\widehat h-h$. Fractional-frequency amplitudes are dimensionless, so the PSD has units $\mathrm{{Hz}}^{{-1}}$. The curves use the same one-sided periodogram convention.}}\label{{fig:psd}}\end{{figure*}}

\subsection{{Phase recovery}}\label{{subsec:PhaseRecoveryAlignment}}
For each channel, define $z_c=h_c+i\mathcal H[h_c]$ and $\widehat z_c=\widehat h_c+i\mathcal H[\widehat h_c]$. At the clean-signal envelope peak $j_{{\star,c}}$, the phase error is
\begin{{equation}}
\Delta\phi_c=\arg[\widehat z_{{c,j_{{\star,c}}}}z_{{c,j_{{\star,c}}}}^*].
\end{{equation}}
The mean absolute phase errors are ${primary['phase_A']['mean']:.2f}^\circ$ in $A$ and ${primary['phase_E']['mean']:.2f}^\circ$ in $E$. The corresponding medians are ${primary['phase_A']['median']:.2f}^\circ$ and ${primary['phase_E']['median']:.2f}^\circ$, and the 95th percentiles are ${primary['phase_A']['p95']:.2f}^\circ$ and ${primary['phase_E']['p95']:.2f}^\circ$. This diagnostic evaluates the analytic phase of the channel waveform at fixed time; quasinormal-mode phase parameters require a separate mode decomposition.
\begin{{figure*}}[t]\centering\includegraphics[width=\textwidth]{{latest_phase.pdf}}
\caption{{(a) Absolute phase-error distributions for all 6,000 sources in each channel. (b) Analytic phase of the $A$ channel for source {cases[1]['seed']}, also shown in Fig.~\ref{{fig:examples}}(b). The displayed clean-signal envelope exceeds 10\% of its peak. The common unwrapped phase branch preserves the relative phase modulo $2\pi$; the arrival time and phase remain fixed. Shading marks the gap and the dotted line marks the clean envelope peak.}}\label{{fig:phase}}\end{{figure*}}
'''
replace_block(r'\section{Reconstruction results}',r'\section{Discussion and Conclusions}',results)

discussion=rf'''
\section{{Discussion and Conclusions}}\label{{sec:discussion}}
We have presented DGF for joint denoising and reconstruction of black hole ringdown signals with data gaps. The network combines complex time--frequency features with signed multiscale temporal information and synthesizes the clean $A/E$ waveforms using the observed context and gap mask. Simulations follow standard LISA response and noise models with the assumptions specified in Sec.~\ref{{sec:method}}. On the independent variable-gap test, the mean noise-weighted mismatch is {five(primary['psd']['mean'])}; the $\rho\ge10$ subset has mean {five(strong['psd']['mean'])}. Mean absolute phase errors are ${primary['phase_A']['mean']:.2f}^\circ$ and ${primary['phase_E']['mean']:.2f}^\circ$ in the two channels.

The complete distributions and conditional diagnostics describe the accuracy across the tested source population. Successful examples recover intervals containing approximately 39\%--43\% of the signal squared norm. The peak-inside-gap subset has mean mismatch {five(peak['psd']['mean'])}, compared with {five(outside['psd']['mean'])} for peaks outside the gap. Full-waveform and gap-local residuals quantify amplitude recovery alongside the normalized overlap. The fixed-duration tests in Appendix~\ref{{app:fixed_gaps}} use the same sources and noise and provide additional checks over the trained gap range.

The simulations cover nonspinning progenitor binaries, isotropic orientations, an idealized equal-arm constellation and stationary Gaussian instrument noise. The analysis uses event-centered 528-s windows and single 32--52-s gaps. Extending the model to moving unequal-arm constellations, nonstationary noise, confusion foregrounds, overlapping sources and multiple gaps calls for simulations of those conditions. Reconstruction uncertainty and variation between trained estimators can be assessed with probabilistic models and repeated training runs.

DGF provides a clean-waveform point estimate for waveform inspection and initial estimates in ringdown analysis. Parameter inference uses the original observation likelihood, constructed from $d-Gh(\theta)$ and the noise covariance. Posterior bias and coverage tests would quantify the usefulness of reconstruction in that setting. Bayesian gap treatment~\cite{{Baghi2019}} and learned denoising~\cite{{Wang:2024waveformer}} address complementary parts of this analysis. Fitting individual mode frequencies and damping times would connect reconstruction fidelity to black hole spectroscopy.
'''
replace_block(r'\section{Discussion and Conclusions}',r'\begin{acknowledgments}',discussion)

replace_block(r'\section{Controlled waveform and noise prescription}',r'\bibliography{DGF_references}',rf'''
\section{{Additional waveform diagnostics}}\label{{app:complementary}}
Figure~\ref{{fig:snr_gap}} shows every independent test source as a function of network SNR and separates the distributions by gap location. Both panels use the primary variable-gap predictions from Sec.~\ref{{sec:results}}. The $\rho\ge10$ summaries retain {strong['n']} of the 6,000 test sources; the complete-population results retain all sources.
\begin{{figure*}}[t]\centering\includegraphics[width=\textwidth]{{latest_snr_gap.pdf}}
\caption{{Additional diagnostics for the same independent test population. (a) Joint noise-weighted mismatch versus network SNR, with the dotted line at $\rho=10$. (b) Conditional mismatch distributions according to whether the joint signal-envelope peak falls inside the gap. All sources contribute to the complete-population statistics.}}\label{{fig:snr_gap}}\end{{figure*}}

Figure~\ref{{fig:complementary_examples}} gives sources at ranks 3000, 5400 and 6000 in ascending noise-weighted mismatch. Their mismatches are {', '.join(five(c['mismatch']) for c in s['diagnostic_cases'])}. The largest mismatch in this population is {primary['psd']['max']:.5f}. These cases complement the successful illustrations in the main text and retain the full error range.
\begin{{figure*}}[t]\centering\includegraphics[width=\textwidth]{{latest_diagnostic_examples.pdf}}
\caption{{Waveform reconstructions at the median, 90th-percentile and largest-mismatch ranks of the 6,000-source variable-gap test. The left and right columns show $A$ and $E$. Each source uses one common clean-signal amplitude normalization for both channels and predictions. Shading marks the gap.}}\label{{fig:complementary_examples}}\end{{figure*}}

The phase statistics in Sec.~\ref{{subsec:PhaseRecoveryAlignment}} contain one value per channel and source. This convention preserves source counts when reporting channel-specific means. In intervals containing very little clean signal, gap-local normalization by $\|mh\|_2$ magnifies residuals. The complete-population amplitude diagnostic uses the full target norm, while the main-text gap-local statistic is reported for sources whose joint envelope peaks are inside the interval.

\section{{Fixed-duration gaps and reference comparison}}\label{{app:fixed_gaps}}
We evaluate 33- and 50-s gaps on the same 6,000 source waveforms and noise realizations, preserving each gap's center. Table~\ref{{tab:fixed_gaps}} gives the joint noise-weighted mismatch of the selected DGF model. These tests vary the mask while holding the physical source population and noise fixed.
\begin{{table}}[t]\centering
\caption{{Additional gap-duration tests on the same independent sources. Each complete-population row contains 6,000 sources; each $\rho\ge10$ row contains {strong['n']} sources.}}\label{{tab:fixed_gaps}}
\begin{{tabular}}{{lrrr}}\hline\hline
Gap and SNR & Mean $\mathcal M$ & Median $\mathcal M$ & \shortstack{{95th\\percentile}}\\\hline
33 s / all & {five(s['fixed_duration_results']['fixed66']['psd']['mean'])} & {five(s['fixed_duration_results']['fixed66']['psd']['median'])} & {five(s['fixed_duration_results']['fixed66']['psd']['p95'])}\\
33 s / $\rho\ge10$ & {five(s['fixed_duration_results']['fixed66']['snr_ge10']['mean'])} & {five(s['fixed_duration_results']['fixed66']['snr_ge10']['median'])} & {five(s['fixed_duration_results']['fixed66']['snr_ge10']['p95'])}\\
50 s / all & {five(s['fixed_duration_results']['fixed100']['psd']['mean'])} & {five(s['fixed_duration_results']['fixed100']['psd']['median'])} & {five(s['fixed_duration_results']['fixed100']['psd']['p95'])}\\
50 s / $\rho\ge10$ & {five(s['fixed_duration_results']['fixed100']['snr_ge10']['mean'])} & {five(s['fixed_duration_results']['fixed100']['snr_ge10']['median'])} & {five(s['fixed_duration_results']['fixed100']['snr_ge10']['p95'])}\\
\hline\hline\end{{tabular}}\end{{table}}

A reference DGF configuration with joint-channel synthesis and the training-derived signal basis is evaluated on the identical test observations. It uses the preceding fitted synthesis parameters and omits the final noise-conditioned head. For variable gaps, its mean mismatch is {five(s['same_test_reference']['variable']['previous']['psd']['mean'])}. The selected model has paired mean difference ${sci(s['same_test_reference']['variable']['paired_delta']['mean'])}$, with 95\% bootstrap interval $[{sci(s['same_test_reference']['variable']['paired_delta']['ci95'][0])},{sci(s['same_test_reference']['variable']['paired_delta']['ci95'][1])}]$. The interval uses 5,000 paired resamples of the independent test sources. This comparison concerns the complete fitted configurations, including their synthesis parameters and coefficient heads.

''')

# Use conventional terminology consistently; the observation equation defines
# how noise is represented inside the modeled gap.
text=text.replace('signal-suppression interval','gap').replace('signal-suppression intervals','gaps')
text=text.replace('suppression intervals','gaps').replace('suppressed interval','gap')
text=text.replace('phenomenological benchmark','simulation').replace('numerical benchmark','test set')
text=text.replace('The direct mismatch and full-window optimal signal norm are', 'The joint-channel mismatch and optimal network signal-to-noise ratio (SNR) are')
text=re.sub(r'\\mathcal M(?![_A-Za-z])',lambda m:r'\mathcal M_{AE}',text)
text=text.replace(r'\begin{table}[t]\centering',r'\begin{table}[t]\centering\setlength{\tabcolsep}{2pt}')
assert 'four-component' not in text and 'noise family' not in text
assert '0.00537' not in text and '202624' not in text and '0.02715' not in text
assert not any('{'+n+'}' in text for n in ['examples_best_peak.pdf','mismatch_cdf_presented.pdf','error_psd_presented.pdf','phase_presented.pdf','examples.pdf','phase_and_R.pdf','training.pdf'])
labels=set(re.findall(r'\\label\{([^}]+)\}',text))
refs=set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',text))
assert not (refs-labels), sorted(refs-labels)
p.write_text(text,encoding='utf-8',newline='\n')
(HERE/'unified_manuscript_checks.json').write_text(json.dumps({
 'selected_checkpoint_sha256':s['checkpoint_sha256'],'all_reconstruction_results_same_source_population':True,
 'independent_test_sources':6000,'old_benchmark_values_removed':True,
 'old_results_figures_referenced':False,'abstract_word_count':len(text.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0].split()),
 'source_seed_range':s['seeds'],'undefined_internal_references':sorted(refs-labels),
 'standard_terms':['ringdown','waveform reconstruction','noise-weighted mismatch','network signal-to-noise ratio','time-delay interferometry'],
},indent=2))
print(json.dumps({'bytes':len(text.encode()),'abstract':text.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]},indent=2))
