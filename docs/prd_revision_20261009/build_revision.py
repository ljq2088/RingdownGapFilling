"""Compose the PRD revision from the downloaded, authoritative cloud source.
Retains title, authors, Introduction context, DGF identity, and four-section plan.
Numerical tables are rendered only from the independent evaluation JSON.
"""
from pathlib import Path
import json, re, shutil, difflib

root=Path(__file__).resolve().parent
before=root/'source_before';out=root/'source_edited';out.mkdir(exist_ok=True)
source=(before/'DGF.tex').read_text(encoding='utf-8')
ev=json.loads((root/'evaluation/evaluation.json').read_text())
prefix=source.split(r'\begin{abstract}')[0]
# REVTeX controls captions; unnecessary caption/wrapfig/changes packages removed.
prefix=re.sub(r'^\\usepackage(?:\[[^\]]*\])?\{(?:caption|wrapfig|changes|placeins|float|subfig|threeparttable)\}[^\n]*\n','',prefix,flags=re.M)
prefix='\n'.join(x for x in prefix.splitlines() if not x.lstrip().startswith('%'))+'\n'
prefix=prefix.replace('[prd,aps,a4paper,','[10pt,prd,aps,a4paper,').replace(r'\allowdisplaybreaks','')
prefix=re.sub(r'\\author\{([^{}]+)\\footnote\{E-mail:\s*([^{}]+)\}\}',r'\\author{\1}\\email{\2}',prefix)
intro=source[source.index(r'\section{Introduction}'):source.index('Here we focus')]
intro='\n'.join(x for x in intro.splitlines() if not x.lstrip().startswith('%'))

abstract=r'''\begin{abstract}
Data gaps can remove informative cycles from a rapidly decaying black hole ringdown. We present DenoiseGapFiller (DGF), a dual-branch encoder--decoder that combines complex time--frequency features with multiresolution temporal context to estimate the complete, clean waveform. We extend this framework with an explicit gap-mask residual module and a phase-sensitive frequency-domain training objective. In a controlled, ringdown-inspired synthetic benchmark, the mean direct noise-weighted mismatch on all 1000 prospectively generated test examples decreases from $2.71\times10^{-2}$ for the original checkpoint to $5.37\times10^{-3}$; the median is $1.52\times10^{-3}$ and the 95th percentile is $1.01\times10^{-2}$. The direct time-domain mean mismatch is $5.04\times10^{-3}$. The gain persists for gaps covering the waveform-envelope peak, although these cases have a larger error tail. We report paired comparisons, amplitude and phase diagnostics, and sensitivity to the spectral window. The results establish improved waveform recovery within the specified signal-masking benchmark. The waveform and noise prescriptions are phenomenological, and detector-calibrated LISA/TDI performance and astrophysical parameter inference remain to be validated.
\end{abstract}
\maketitle
'''
intro+=r'''
Here we focus on the joint denoising and reconstruction of ringdown-like oscillations when a short interval of the signal is suppressed. DGF combines a complex time--frequency branch with a multiresolution time-domain branch. The first supplies local oscillatory features, including phase, while the second retains signed temporal context. Their fusion is a design motivation; its superiority over independently retrained single-branch networks is not assumed. An explicit gap mask and a residual refinement module make the location of the interruption available to the learned mapping. Complex Fourier residuals penalize phase as well as amplitude errors.

We assess this ringdown-oriented framework on a complete prospective test population, retaining the original DGF architecture and benchmark lineage. Direct waveform, amplitude, and effective phase diagnostics include cases with masked envelope peaks. The benchmark suppresses the signal while retaining noise; it neither removes all recorded samples nor supplies calibrated detector data. These distinctions bound the interpretation of the results.

Section~\ref{sec:method} specifies the benchmark, representations, refinement, and selection protocol. Section~\ref{sec:results} gives the complete test results and complementary diagnostics. Section~\ref{sec:discussion} discusses their significance and physical limitations. Appendixes~\ref{app:model_analysis} and~\ref{app:sequence_modeling} provide implementation and benchmark details.
'''
method=r'''
\section{Methodology}\label{sec:method}
\subsection{Noise model}\label{noise model}
The motivating application is space-based gravitational-wave analysis. An analytic sensitivity curve, such as the LISA model of Ref.~\cite{robson2019construction}, provides useful physical context, but is not the PSD used by the present generator. The benchmark instead starts from a supplied three-series noise file, forms $n_A=(n_Z-n_X)/\sqrt{2}$, and estimates its empirical PSD by Welch averaging with a 15-s sampling interval and 1024-sample segments. The reference file does not record sufficient metadata to establish its units, TDI generation, or complete provenance; this algebraic channel combination alone does not validate it as a calibrated LISA observable.

Stationary Gaussian noise is synthesized by coloring the Fourier transform of independent white Gaussian samples with an interpolated amplitude spectrum. The source spectrum extends only to $1/30\,\mathrm{Hz}$, whereas the benchmark is sampled at 2 Hz; the implementation fills frequencies outside the reference range with a PSD of $10^{-40}$ in its numerical units. Its Fourier normalization also scales the synthesized PSD by $2/N$ relative to the coloring spectrum, for a parent sequence of length $N$. Consequently, the input coloring spectrum cannot be used as a physical detector sensitivity or as the actual PSD of the cropped noise. These choices are retained to evaluate the existing trained checkpoints on their benchmark distribution, and their discrete implementation is specified in Appendix~\ref{app:sequence_modeling}.

For evaluation we use $S_{\rm cal}(f)$ estimated from 200 independently reserved, noise-only benchmark segments, with the same 1056-sample duration and Fourier grid as the evaluated waveforms. No final-test waveform or noise is used to construct this weighting. Calibration characterizes the simulated process; it does not restore the missing detector calibration. Throughout, amplitudes and PSDs are expressed in benchmark numerical units. We use $R=\|h\|_2/\|n\|_2$ only as a signal-to-noise amplitude ratio, and do not identify it with a matched-filter or network SNR.

\subsection{Ringdown signal benchmark}
The physical motivation is the damped-oscillation structure of a black hole ringdown. In a calibrated Kerr model, a mode has the form~\cite{berti2006gravitational,berti2009qnm}
\begin{equation}
h_a(u)=C_a e^{-u/\tau_a}\cos(\omega_a u+\phi_a),\qquad u\geq0,
\end{equation}
where the mode frequencies and damping times depend on the remnant mass and spin, and $C_a$ includes excitation, angular, distance, and detector factors. This expression motivates the reconstruction task; the numerical benchmark is generated by the legacy frequency-domain prescription detailed below, rather than by a newly calibrated realization of this physical model.

Four oscillatory components retain the legacy labels $(2,2)$, $(2,1)$, $(3,3)$, and $(4,4)$. The generator draws independent uniform coordinates $M\in[10^5,10^6]$, $q\in[0.5,0.8]$, and $z\in[4,6]$, with fixed orientation and phase conventions. The mass coordinate enters the generator through $M T_\odot$, and $z$ sets a distance scale. They are waveform-family coordinates with solar-mass and redshift inspired labels, rather than independently validated astrophysical source parameters. In particular, the implementation does not consistently distinguish total, remnant, source-frame, and detector-frame mass.

Writing $\eta=q/(1+q)^2$ and $\chi=\eta(2\sqrt{3}-3.5171\eta+2.5763\eta^2)$, the implemented frequencies and decay times are
\begin{align}
\omega_a^{\rm b}&=\frac{f_{1a}+f_{2a}(b_a-\chi)^{f_{3a}}}{M T_\odot},\label{eq:benchmark_frequency}\\
\tau_a^{\rm b}&=\frac{2[q_{1a}+q_{2a}(1-\chi)^{q_{3a}}]}{\omega_a^{\rm b}},
\end{align}
with $b_a=(1,1,3,4)$ in the component order above. Table~\ref{tab:frequency_coefficients} records the coefficients. The standard Kerr fits use $b_a=1$ for every mode~\cite{Berti_2006}; the third and fourth components of this retained benchmark therefore are not faithful Kerr QNMs. The labels identify components of the legacy generator and do not establish black hole spectroscopy accuracy.
\begin{table}[t]
\caption{Coefficient values retained by the benchmark. The factors $b_a$ in Eq.~\eqref{eq:benchmark_frequency} differ from the standard Kerr prescription for the last two rows.}\label{tab:frequency_coefficients}
\centering\begin{tabular}{crrrrrr}\hline\hline
Label & $f_1$ & $f_2$ & $f_3$ & $q_1$ & $q_2$ & $q_3$\\\hline
$22$ &1.5251&-1.1568&0.1292&0.7000&1.4187&-0.4990\\
$21$ &0.6000&-0.2339&0.4175&-0.3000&2.3561&-0.2277\\
$33$ &1.8956&-1.3043&0.1818&0.9000&2.3430&-0.4810\\
$44$ &2.3000&-1.5056&0.2244&1.1929&3.1191&-0.4825\\\hline\hline
\end{tabular}\end{table}
The implemented component multipliers are
\begin{align}
A_{22}^{\rm b}&=0.303+0.571\eta,& A_{21}^{\rm b}&=0.060+0.099\eta,\nonumber\\
A_{33}^{\rm b}&=0.671+0.157\eta,& A_{44}^{\rm b}&=0.122-0.188\eta-0.964\eta^2.
\end{align}
They are numerical coefficients of this benchmark. Their historical association with EMOP energy fits does not make them calibrated strain amplitudes: the energy-to-strain conversion and the odd-mode symmetry factors are absent, and the $21$ and $33$ coefficient order differs from the cited fits~\cite{Baibhav2018energy}. Thus the reconstruction results concern the actual simulated family, not the accuracy of this excitation prescription. Fixed angular factors and a TianQin-inspired modulation are applied in frequency space, followed by a discrete inverse transform, a cyclic shift, a 1056-sample crop, and an amplitude factor of 0.03. No finite-arm LISA signal response or link-level TDI processing is applied. Appendix~\ref{app:sequence_modeling} gives the discrete data contract.

\subsection{Data gaps}\label{sec:gaps}
We define a binary window $G_j$ equal to zero in the masked interval and one elsewhere, and use the observation
\begin{equation}d_j=G_j h_j+n_j,\qquad m_j=1-G_j.\label{eq:observation}\end{equation}
The target is the complete, clean $h$. Noise remains present inside the gap, so this is a controlled signal-suppression experiment. Actual missing recorded samples would instead be represented by $G(h+n)$ and require a corresponding change to the input and noise likelihood. The present results do not establish performance in that observation model.

Each example has 1056 samples at 2 Hz, corresponding to a 528-s segment. A single rectangular mask spans 66 samples (33 s, or 6.25\% of the segment). Validation and test masks start uniformly at integer samples 132--263, corresponding to 66--131.5 s from the segment start. No mission duty cycle or Poisson interruption process is simulated. Since informative power is concentrated near the signal peak, we separately evaluate masks containing $j_\star=\arg\max_j|h_j+i\mathcal H[h]_j|$, where $\mathcal H$ denotes the Hilbert transform. This grouping is a diagnostic based on the clean target and is never supplied to inference.

\subsection{DGF reconstruction pipeline}
DenoiseGapFiller retains the dual-branch encoder--decoder of the original framework. The time--frequency branch uses a complex Morlet continuous wavelet transform (CWT), represented by its modulus and argument as two explicit channels. It is not a constant-$Q$ transform, despite the historical class name. Sixteen linearly spaced scales and a factor-four temporal subsampling give a $2\times16\times264$ input. Learned temporal and scale mixing and three convolutional stages map it to $8\times32\times64$ features.

The temporal branch normalizes the observation using its own mean and standard deviation, performs an eight-level db4 discrete wavelet transform, and reconstructs eight time-domain subband signals. It retains the coarsest approximation and seven detail components; the complete coefficient set is not retained. The representation is therefore not claimed to be an exactly invertible encoding of the input. Each subband is divided into 32 patches of 64 samples with stride 32, explaining the length $1056=64+31\times32$. This is an architectural choice rather than a physical ringdown timescale.

The two $8\times32\times64$ feature arrays are added before token, convolutional, and sinusoidal positional embeddings. A residual two-dimensional convolution precedes 16 sequence blocks. Each block applies channel-wise scaled dot-product attention directly to the current features, followed by residual normalization and a feedforward layer. The evaluated implementation does not call learned multihead query/key/value projections. The decoder applies an MLP, a learned linear map to 64-sample patches, overlap averaging, a sum over the synthesized subband channels, and a final one-dimensional convolution. Input normalization is then restored. The linear decoder map and subband sum are learned reconstruction operations, not mathematical inverses of all encoding steps.

\begin{figure*}[t]
\centering\includegraphics[width=0.98\textwidth]{architecture_prd.pdf}
\caption{Checkpoint-specific DGF pipeline and residual refinement. Both base branches use only the observation; the refinement additionally receives the known gap mask. The clean target appears only in training and evaluation. CWT phase is explicitly represented, while decoder operations synthesize the output. Array sizes exclude the batch dimension.}\label{fig:pipeline}\label{fig:architecture}
\end{figure*}

The updated predictor adds a residual module to the frozen original DGF checkpoint:
\begin{equation}\widehat h=b(d)+0.2s\,r_\psi(d/s,b(d)/s,m,t),\label{eq:refinement}\end{equation}
where $b(d)$ includes both DGF branches, $s$ is the RMS of the observation with a numerical floor, and $t$ is a coordinate linearly spaced on $[-1,1]$. The module consists of an input convolution, six residual dilated convolutional blocks of width 24 and kernel size 7, with dilations $1,2,4,8,16,32$, group normalization and GELU, and an output convolution. Its zero-initialized output starts from the original prediction. There are 25,489 trainable refinement parameters, added to 9,184,114 parameters in the stored base model. Only the refinement is optimized in the new stages; the complete model is not retrained end to end. RMS normalization supplies numerical conditioning and is not PSD whitening.

\subsection{Loss function and training strategy}\label{loss}
The refinement objective combines relative waveform error, an increased penalty in the masked interval, a complex spectral residual, and direct spectral mismatch:
\begin{equation}\begin{aligned}
\mathcal L={}&\frac{\sum_j(1+2m_j)(\widehat h_j-h_j)^2}{\sum_jh_j^2}
+\frac12\frac{\sum_k w_k|\widehat H_k-H_k|^2}{\sum_k w_k|H_k|^2}\\
&+\frac14\left[1-\frac{\sum_k w_k\operatorname{Re}(\widehat H_k H_k^*)}{\sqrt{\sum_k w_k|\widehat H_k|^2\sum_k w_k|H_k|^2}}\right].\label{eq:loss}
\end{aligned}\end{equation}
$H$ and $\widehat H$ are the discrete Fourier transforms of the target and prediction, and $w_k\propto1/S_{\rm cal}(f_k)$, normalized to unit average; DC and Nyquist have zero weight. Unlike a magnitude-only loss, the complex residual penalizes phase differences. Numerical denominator floors in scaled benchmark units are recorded in Appendix~\ref{app:model_analysis}. The objective supervises the full sequence, including denoising outside the mask.

The refinement uses 12,000 unique training waveforms and their noise realizations, with disjoint seed ranges for 1500 validation examples. An initial stage uses two masks per training waveform. The augmented stage warm-starts from that stage and pairs each waveform with six train-only noise permutations and masks, giving 72,000 combinations rather than 72,000 independent sources. Half the views use uniformly placed masks; the other half place a mask near the target-envelope peak, clipping its start to the same 132--263 interval. This targets a difficult region during training without filtering the natural test population. No validation or test noise enters these recombinations.

AdamW uses batch size 128 and weight decay $10^{-4}$. The augmented stage has cosine learning-rate cycles of 200 epochs between $10^{-4}$ and $2\times10^{-6}$. Checkpoints are selected using $0.75$ times the validation population mean PSD mismatch plus $0.25$ times the mean for validation peaks inside the gap. A minimum of 400 epochs, patience of 250 epochs, and an improvement threshold of $10^{-5}$ define early stopping. The run reached its limit of 1200 epochs; epoch 1095 was selected. A fresh final set of 1000 waveforms was generated only after this selection. Historical base-model split indices were not archived; the final test is a prospective sample from the retained generator, with all rows included in the comparison.
\begin{figure}[tbp]\centering\includegraphics[width=\columnwidth]{training.pdf}
\caption{Validation population mismatch and the composite selection score during the augmented refinement stage. The dashed line identifies the selected checkpoint. The curves have different population weights; their separation is not a training--validation generalization gap.}\label{fig:loss}\end{figure}
'''

def row(name):
    d=ev['groups']['all'][name]
    return f"{ {'baseline':'Original DGF','previous':'First refinement','augmented':'Augmented refinement','interpolation':'Linear interpolation'}[name]} & {d['time_mismatch']['mean']:.5f} & {d['psd_mismatch']['mean']:.5f} & {d['psd_mismatch']['median']:.5f} & {d['psd_mismatch']['p95']:.5f} \\\\\n"
table=r'''\begin{table*}[t]\centering
\caption{Direct reconstruction mismatch for all 1000 final-test examples. No waveform is excluded and no time or phase alignment is fitted. Mean values are arithmetic means over examples. Interpolation acts on the noisy observed context and is not a joint denoising baseline.}\label{tab:results}
\begin{tabular}{lrrrr}\hline\hline
Predictor & Mean $\mathcal M_t$ & Mean $\mathcal M_S$ & Median $\mathcal M_S$ & 95th percentile $\mathcal M_S$\\\hline
'''+''.join(row(n) for n in ['baseline','previous','augmented','interpolation'])+r'''\hline\hline\end{tabular}\end{table*}
'''
results=r'''
\section{Results}\label{sec:results}
We compare the original DGF checkpoint, the first refinement, and the selected augmented refinement on identical final-test observations and clean targets. The test seeds are 2026240000--2026240999, disjoint from the registered refinement training and validation ranges. All 1000 examples, including failures, contribute to the primary statistics. Linear interpolation replaces masked samples using neighboring noisy observations and leaves the observed samples unchanged. It measures the behavior of a simple gap bridge, but cannot separate the benefits of denoising from those of reconstruction.

\subsection{Direct waveform mismatch}\label{subsec:Mismatch}
The time-domain normalized overlap and mismatch are
\begin{equation}
\mathcal M_t=1-\frac{\sum_jh_j\widehat h_j}{\sqrt{\sum_jh_j^2\sum_j\widehat h_j^2}}.
\end{equation}
For a noise-weighted comparison we define
\begin{align}
(a|b)_S&=4\Delta f\operatorname{Re}\sum_{k=1}^{T/2-1}\frac{\widetilde a_k\widetilde b_k^*}{S_{\rm cal}(f_k)},\\
\mathcal M_S&=1-\frac{(h|\widehat h)_S}{\sqrt{(h|h)_S(\widehat h|\widehat h)_S}},\label{eq:mismatch}
\end{align}
where $\widetilde a_k=\Delta t\,\mathrm{rFFT}(a)_k$, $T=1056$, $\Delta t=0.5$ s, and $\Delta f=1/(T\Delta t)$. The primary metric uses the rectangular segment and excludes DC and Nyquist. The shared normalization constants cancel in the overlap. Neither metric is maximized over a relative time shift or phase. With signed real overlaps, mismatch lies in $[0,2]$ and can exceed one for an anticorrelated reconstruction. A low PSD increases a frequency bin's relative weight; it does not guarantee a high normalized match. A constant rescaling of the entire PSD leaves Eq.~\eqref{eq:mismatch} unchanged.

'''+table+r'''
The augmented refinement reduces the population mean $\mathcal M_S$ by 80.2\% relative to the original DGF checkpoint and by 37.8\% relative to the first refinement. The paired mean differences are $-0.02178$ and $-0.003257$, with 95\% percentile bootstrap intervals $[-0.02708,-0.01683]$ and $[-0.00531,-0.00132]$, respectively. These intervals use 5000 paired resamples of the test examples and quantify sampling variation for the fixed fitted checkpoints; they are not estimates of training-seed variability. Figure~\ref{fig:mismatch} shows the complete empirical distributions. The mean, median, and upper percentiles characterize different aspects of the error tail and should not be interchanged.
\begin{figure*}[t]\centering\includegraphics[width=\textwidth]{mismatch_cdf.pdf}
\caption{Empirical distributions of direct time-domain and calibrated-noise-weighted mismatch for the same 1000 test examples. Each curve includes every example; the logarithmic horizontal axis resolves the central distribution and the failure tail. Interpolation retains noise outside the mask.}\label{fig:mismatch}\end{figure*}

\subsection{Gap location and difficult cases}
The clean-envelope peak lies inside the gap for 121 examples. In this subgroup the mean $\mathcal M_S$ is 0.14016 for original DGF, 0.04580 for the first refinement, and 0.02771 for the augmented refinement. The latter has median 0.00320 and 95th percentile 0.11283. The paired mean difference relative to the first refinement is $-0.01809$, with 95\% bootstrap interval $[-0.03491,-0.00252]$. For the remaining 879 examples, the augmented mean is 0.00229. Thus an improvement in the complete population coexists with substantially harder peak-masking cases.

The augmented model's worst direct PSD mismatch is 1.01221, and its maximum absolute peak-phase deviation is $130.09^\circ$. Such examples are retained rather than trimmed. Figure~\ref{fig:examples} displays examples nearest the median, 90th percentile, and maximum augmented mismatch. Selection by these declared ranks exposes a central case and failure modes; it does not provide independent statistical evidence. No inference of missing information is unique merely because a reconstruction appears smooth.
\begin{figure*}[t]\centering\includegraphics[width=\textwidth]{examples.pdf}
\caption{Central, upper-tail, and worst reconstructions from the fixed test population, selected at ranks 500, 900, and 1000 after sorting the augmented mismatch. All panels show the same target and original/refined outputs; gray shading identifies the 33-s signal mask. The plotted amplitudes use the generator's numerical units.}\label{fig:examples}\end{figure*}

\subsection{Amplitude and spectral diagnostics}\label{subsec:MeanSquaredError}\label{subsec:AmplitudeDensityFrequencySpectrum}
Normalized overlap is insensitive to an overall positive amplitude factor. We therefore also report $E=\|\widehat h-h\|_2/\|h\|_2$. Its augmented mean is 0.06496, median 0.04753, and 95th percentile 0.14741; the original DGF mean is 0.13370. The mean signed norm bias, $\|\widehat h\|_2/\|h\|_2-1$, is $-0.00200$, but cancellation makes this signed quantity insufficient on its own.

Inside the 121 peak-containing masks, $\|m(\widehat h-h)\|_2/\|mh\|_2$ has augmented mean 0.12349 and median 0.05735, compared with an original-model mean of 0.33401. This mask-local relative measure becomes ill conditioned when the target is almost zero inside a late gap. Its full-population mean is 12.12 and is not a claim of uniform local reconstruction accuracy. Per-example local errors and their normalizations are retained in the evaluation archive, alongside gap error normalized by the complete target norm.

Figure~\ref{fig:psd} compares the ensemble mean periodogram of the prediction error $\widehat h-h$ with that of the simulated noise. This diagnoses frequency-dependent errors rather than the restoration of a physical QNM peak. A small broadband mismatch does not guarantee accurate recovery of every weak component.
\begin{figure}[tbp]\centering\includegraphics[width=\columnwidth]{error_psd.pdf}
\caption{Mean one-sided periodograms of the simulated noise and of original/refined prediction errors over all 1000 test examples. Units are benchmark amplitude squared per hertz. The error spectrum is a diagnostic and is not a detector sensitivity curve or a mode-resolution test.}\label{fig:psd}\end{figure}

As a sensitivity check, a periodic Hann window is applied to both waveforms and paired with a noise PSD calibrated using the same window. Mean mismatches are 0.04736, 0.01647, and 0.01116 for original DGF, first refinement, and augmented refinement. The ranking is retained, but the numerical values differ substantially from the primary rectangular-window metric because windowing changes the relative emphasis on parts of the finite segment. We report this dependence rather than choosing the window that produces the smallest score.

\subsection{Phase recovery and alignment}\label{subsec:PhaseRecoveryAlignment}
Let $z=h+i\mathcal H[h]$ and $\widehat z=\widehat h+i\mathcal H[\widehat h]$. At the target-envelope peak $j_\star$, we define the wrapped phase diagnostic
\begin{equation}\Delta\phi_\star=\arg[\widehat z_{j_\star}z_{j_\star}^*],\qquad |\Delta\phi_\star|\in[0,\pi].\end{equation}
No phase rotation, time alignment, or fit to the clean target is performed. This is the effective phase of the composite analytic waveform and not an estimate of each individual QNM phase. The augmented mean absolute deviation is $1.50^\circ$, median $0.73^\circ$, 90th percentile $2.69^\circ$, and 95th percentile $4.08^\circ$. The original-model mean is $4.32^\circ$. For peaks inside the mask, the augmented mean is $5.20^\circ$ and the 95th percentile is $19.93^\circ$. The distribution in Fig.~\ref{fig:phase} shows why a small signed average or central error does not imply uniformly precise recovery.
\begin{figure*}[t]\centering\includegraphics[width=\textwidth]{phase_and_R.pdf}
\caption{Left: complete absolute peak-phase distributions, with a linear central region and logarithmic scale above $0.2^\circ$. Right: direct PSD mismatch against the numerical amplitude ratio $R$ for every test example. $R$ is not matched-filter SNR. Composite Hilbert phase is a waveform diagnostic, not a measurement of separate mode phases.}\label{fig:phase}\end{figure*}

\subsection{Refinement ablation}
A separate matched-budget experiment compares refinement using a time-domain objective alone, adding the explicit mask, adding spectral terms, and adding both. It uses 2048 unique training waveforms with two masks each, the same 1500 validation examples, three initialization seeds per variant, and at most 100 epochs with patience 20. The composite validation scores, averaged over the three seeds, are 0.03004, 0.02657, 0.02719, and 0.02440, respectively; the mask-plus-spectrum variant performs best in each seed. This supports the refinement design under the stated small-budget protocol. It is not a retrained ablation of the two original DGF branches, and the long augmented run has not been replicated across three seeds. The final test set is not used to choose the ablation variant.
'''
discussion=r'''
\section{Discussion and Conclusions}\label{sec:discussion}
We have presented DGF for joint denoising and reconstruction of ringdown-inspired signals and extended it with explicit gap information and phase-sensitive residual refinement. The two original branches retain complementary complex time--frequency and signed multiresolution features. The added module changes a frozen waveform estimate using the observation and known signal mask, preserving the architecture and reconstruction task of the original study.

The complete prospective test population supports improved direct waveform agreement within this benchmark. The mean PSD mismatch decreases from 0.02715 to 0.00537, with a paired confidence interval excluding zero for the improvement. Amplitude and effective phase diagnostics independently show lower average errors. The gain persists for peak-containing masks and under a different spectral window. However, the peak-masked subgroup and worst examples retain pronounced tails, so population-average performance cannot be interpreted as reliable recovery for every interruption. The reported original-versus-refined comparisons do not isolate the contribution of each change in training, augmentation, and warm-start history; the smaller matched-budget ablation provides only partial evidence about mask and spectral terms.

The scientific motivation is preservation of ringdown oscillatory information for subsequent analysis, but the evidence is at waveform level. The legacy amplitudes, two frequency fits, Fourier normalization, and response/noise conventions preclude calibrated LISA or astrophysical interpretations. The explicit data contract permits reconstruction comparisons without validating a physical model. Consistent waveform normalization, documented noise and detector response, and TDI validation are required before studying individual Kerr modes, detection evidence, or black hole spectroscopy.

Likewise, Eq.~\eqref{eq:observation} removes the signal while leaving noise in the mask. It is useful for a controlled study of prediction from incomplete signal context, but is different from missing telemetry. Multiple gaps, other durations, nonstationary or non-Gaussian noise, random orientations and phases, and overlapping sources are outside the tested distribution. The original branch architecture has not been independently retrained without CWT phase or without its time--frequency branch. Our phase-sensitive inputs are implementation facts; a causal architectural advantage in phase recovery requires those controlled comparisons.

Related methods~\cite{Baghi2019,Blelly2021,Xu_2024,Mao:2025cae,Wang:2024waveformer} use different waveform families or data conditions, so their overlaps cannot provide a controlled ranking with our direct mismatch. Our noisy interpolation reference does not isolate reconstruction from denoising. Fair comparison of accuracy, uncertainty calibration, and cost requires a shared physical benchmark.

A deterministic reconstruction is conditioned on the same observations as any downstream inference and introduces no independent measured information. For signal suppression, a physical likelihood would use the residual $d-Gh(\theta)$ with the appropriate noise covariance. For genuine missing samples, it would use the observed samples and their covariance submatrix. Treating the filled samples as additional independent observations would double-count information and could narrow posteriors without correct coverage. DGF may provide initial estimates or sampling proposals, but benefits in inference must be checked against the same original data, likelihood, and priors. No Bayesian parameter-estimation or posterior-coverage result is claimed here.

The results demonstrate improved prediction within this signal family and identify peak masking as a difficult regime. The framework and evaluation protocol provide a basis for physically calibrated gap-recovery tests, while distinguishing waveform agreement from detector performance and astrophysical inference.
'''
ack=source[source.index(r'\begin{acknowledgments}'):source.index(r'\end{acknowledgments}')+len(r'\end{acknowledgments}')]
appendix=r'''
\appendix
\section{Checkpoint implementation and reproducibility}\label{app:model_analysis}
\subsection{Representations and sequence modeling}
For one example, the CWT input has size $2\times16\times264$, the subband patches have size $8\times32\times64$, and both branch outputs are mapped to that latter layout. Fusion is elementwise addition. For a channel's feature matrix $Z$, the active attention operation is
\begin{equation}A(Z)=\operatorname{softmax}(ZZ^\mathsf T/\sqrt{64})Z.\end{equation}
It is applied independently to each of the eight feature channels. Learned feedforward layers, residual connections, and normalization follow. The allocated multihead-attention parameters are not used in the forward path, so stored parameter counts should not be equated with active attention complexity. Bidirectional here denotes access to the full 32-token sequence and does not denote a causal or streaming model.

The eight-level db4 implementation reconstructs the approximation and detail entries indexed 1--7 of the library's fine-to-coarse coefficient list, omitting entry 0. Cropping restores 1056 samples. Learned decoded channels are summed in time rather than passed to an exact inverse transform of a complete set of original wavelet coefficients. The patch decoder map is initialized from a transposed embedding weight but its trained checkpoint contains independent values; the inference loader clones the shared CPU storage before loading, preserving those values. This addresses a loading discrepancy without changing the trained weights.

\subsection{Numerical conventions}
The refinement operates on arrays multiplied by $10^{22}$ for numerical scale. Its RMS floor is $10^{-4}$. The relative waveform-loss denominator is floored at $10^{-4}$, and the spectral norm denominators at $10^{-5}$ in these scaled units. These are training stabilization choices. The evaluation uses float64 arrays, no such denominator floors for the nonzero evaluated waveforms, and a clipped normalized overlap only to bound floating-point roundoff. Positive amplitude rescaling cancels in mismatch but not in relative residual diagnostics.

Training waveform seeds are 2026101000--2026112999 and validation seeds are 2026121000--2026122499. The augmented initialization seed is 2026100831. Train-only permutations and gap starts are saved in an augmentation manifest. Target-dependent peak placement is used only to construct training masks; inference receives the externally specified mask and observation. The model never receives the clean target, true waveform parameters, or true peak location at inference.

The frozen base checkpoint has SHA-256 prefix \texttt{e7dcf7e55cc6b3f5}; the empirical evaluation PSD archive has prefix \texttt{d9707439dc67af9e}. Full hashes for these and the final predictions, refinement checkpoint, configurations, and history are saved with the evaluation. The code and original research prototype are available at \url{https://github.com/ljq2088/RingdownGapFilling}; this URL identifies the repository and does not assert that the new evaluation artifacts have already been publicly deposited. Exact checkpoint loading, array-level metric recomputation, and the unresolved noise-file provenance must be distinguished when reproducing results.

\section{Legacy benchmark data contract}\label{app:sequence_modeling}
\subsection{Signal discretization}
The frequency-domain function \texttt{sf} is evaluated on \texttt{arange}$(10^{-5},1,10^{-5})$ Hz, containing $K=99999$ entries. The response-modulated sum of its four components is treated as the input to \texttt{irfft} with $N=2(K-1)=199996$, using the library's $1/N$ inverse normalization. The first entry at $10^{-5}$ Hz is assigned the DC position by this operation, and no continuous-Fourier-to-DFT sampling-rate factor is applied. The series is cyclically shifted by $N/2$ samples and cropped from $(N-1056)/2+s$ with a random integer $s\in[0,527]$. The crop is cast to float32 and multiplied by 0.03. This procedure specifies the retained benchmark; it is not asserted to be the correctly normalized inverse transform of a physical strain spectrum.

Fixed angles are $\iota=\pi/4$, $\gamma=\pi/3$, $\theta_d=1.17662$, $\psi_d=-0.873302$, and $\phi_d=-0.615727$ radians, with the legacy TianQin-inspired modulation frequency approximately $1.99\times10^{-5}$ Hz. These do not implement the moving finite-arm LISA response or TDI. The nonspinning binary coordinate used to construct $\chi$ is not an independently measured remnant spin. Physical fits cited for motivation~\cite{Berti_2006,zhang2021parameter,Baibhav2018energy} cannot validate the altered component spectrum or its excitation coefficients.

\subsection{Noise discretization and evaluation weighting}
Let $u_j$ be independent standard Gaussian variates on the parent sequence and let $S_{\rm ref}$ denote the Welch spectrum of the supplied channel combination. The implementation first interpolates this PSD onto the generator grid, then interpolates its square root onto the parent real-FFT grid. Denoting the resulting array by $B_k$, the synthesized sequence is
\begin{equation}n_j^{\rm parent}=\sqrt{f_s}\,\operatorname{irfft}\left[\frac{\operatorname{rfft}(u)_k}{\sqrt N}B_k\right]_j,\qquad f_s=2\,\mathrm{Hz}.
\end{equation}
It is cropped centrally to 1056 samples and cast to float32. Away from endpoint bins, the normalization gives an ensemble one-sided PSD $2B_k^2/N$ before cropping. Window leakage in a short crop further changes the frequency-dependent spectrum. $S_{\rm cal}$ is therefore estimated from independently reserved cropped noise examples on the actual evaluation grid. The Hann sensitivity check uses a periodic Hann window and the correspondingly calibrated windowed-noise PSD.

Small empirical PSD values reflect the synthetic process and its out-of-band fill, not enhanced detector sensitivity. Inverse-PSD weighting emphasizes errors in low-noise bins; normalization does not force overlap toward unity. Population means average per-example mismatches equally, without filtering by mismatch, $R$, or reconstruction quality.
'''
for old,group,reference in [('[-0.02708,-0.01683]','all','baseline'),('[-0.00531,-0.00132]','all','previous'),('[-0.03491,-0.00252]','peak_in_gap','previous')]:
    ci=ev['paired'][group][reference]['ci95']
    results=results.replace(old,f'[{ci[0]:.5f},{ci[1]:.5f}]')
results=results.replace('These intervals use 5000 paired resamples','These intervals use a fixed bootstrap seed (20261009) and 5000 paired resamples')
intro=intro.replace('baghi2019gaps','Baghi2019').replace(',Spadaro:2023hlw','')
method=method.replace('Berti_2006','berti2006gravitational')
appendix=appendix.replace('Berti_2006','berti2006gravitational')
tex=prefix+abstract+intro+method+results+discussion+ack+appendix+'\n\\bibliography{DGF_references}\n\\end{document}\n'
(out/'DGF.tex').write_text(tex,encoding='utf-8')
bib=(before/'DGF_references.bib').read_text(encoding='utf-8')
if 'Baibhav2018energy' not in bib:
    bib+=r'''
@article{Baibhav2018energy,
 author={Baibhav, Vishal and Berti, Emanuele and Cardoso, Vitor and Khanna, Gaurav},
 title={Black Hole Spectroscopy: Systematic Errors and Ringdown Energy Estimates},
 journal={Phys. Rev. D}, volume={97}, pages={044048}, year={2018},
 doi={10.1103/PhysRevD.97.044048}, eprint={1710.02156}, archivePrefix={arXiv}, primaryClass={gr-qc}
}
'''
# Remove mismatched records and the duplicate 2006 Berti entry.
for badkey in ('baghi2019gaps','Spadaro:2023hlw','Berti_2006'):
    bib=re.sub(r'@\w+\s*\{'+re.escape(badkey)+r',.*?(?=\n@|\Z)','',bib,flags=re.S)
bib=bib.replace('Quentin Baghi and John I. Thorpe','Quentin Baghi and James Ira Thorpe')
bib=bib.replace('10.1109/42.730','10.1109/42.7784')
mao=r"""@article{Mao:2025cae,
 author={Mao, Ruiting and Lee, Jeong Eun and Edwards, Matthew C.},
 title={Novel stacked hybrid autoencoder for imputing {LISA} data gaps},
 journal={Phys. Rev. D}, volume={111}, pages={024067}, year={2025},
 doi={10.1103/PhysRevD.111.024067}, eprint={2410.05571}, archivePrefix={arXiv}, primaryClass={gr-qc}
}
"""
bib=re.sub(r'@\w+\s*\{Mao:2025cae,.*?(?=\n@|\Z)',lambda _: mao,bib,flags=re.S)
(root/'bibliography_changes.diff').write_text(''.join(difflib.unified_diff((before/'DGF_references.bib').read_text().splitlines(True),bib.splitlines(True),fromfile='Overleaf original/DGF_references.bib',tofile='PRD revision/DGF_references.bib')),encoding='utf-8')
(out/'DGF_references.bib').write_text(bib,encoding='utf-8')
for p in (root/'evaluation').glob('*.pdf'): shutil.copy2(p,out/p.name)
diff=''.join(difflib.unified_diff(source.splitlines(True),tex.splitlines(True),fromfile='Overleaf original/DGF.tex',tofile='PRD revision/DGF.tex'))
(root/'manuscript_changes.diff').write_text(diff,encoding='utf-8')
labels=re.findall(r'\\label\{([^}]+)\}',tex);refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',tex)
assert all(r in labels for r in refs),set(refs)-set(labels)
citations={c.strip() for group in re.findall(r'\\cite\{([^}]+)\}',tex) for c in group.split(',')}
keys=set(re.findall(r'@\w+\s*\{\s*([^,]+),',bib));assert citations<=keys,citations-keys
print('Manuscript:',len(tex.splitlines()),'lines;',len(tex),'characters; labels/citations verified')
