from pathlib import Path
from zipfile import ZipFile
import re,json,difflib,hashlib

P=Path(__file__).resolve().parent
with ZipFile(P/'overleaf_before.zip') as z:
    original=z.read('DGF.tex').decode('utf-8').replace('\r\n','\n')
    oldbib=z.read('DGF_references.bib').decode('utf-8').replace('\r\n','\n')
s=original; changes=[]
def rep(a,b,tag):
    global s
    assert s.count(a)==1,(tag,s.count(a))
    s=s.replace(a,b);changes.append(tag)
def paragraph(prefix,new,tag):
    a=s.index(prefix)
    b=min(x for x in (s.find('\n\n',a),s.find('\n\\begin{',a)) if x>=0)
    rep(s[a:b],new,tag)

rep('Across the full test population, mean absolute phase errors at the signal-envelope peak are below $4^\\circ$ in both $A/E$ channels.',
'Across the full test population, the mean absolute analytic-phase error at each channel\'s clean-signal envelope peak is below $4^\\circ$.','abstract: channel-specific analytic phase')
rep('The reconstruction problem consequently concerns the phase and decay envelope across the gap, as well as the complete waveform. Gaps covering as much as 20\\% of a short ringdown record illustrate the severity of this problem. Here we study gaps spanning approximately 6\\%--10\\% of a 528-s record.',
'We study 32--52-s gaps in 528-s records, corresponding to approximately 6\\%--10\\% of the analysis window, and assess waveform overlap, amplitude error and phase recovery across the gap.','remove unsupported 20-percent background and generic transition')
rep('A coefficient head conditions a training-derived signal basis on the observed noise level and phase features.',
'A coefficient head adjusts the coefficients of a fixed, training-derived signal basis using observation-derived noise and phase features.','conditioned coefficients, fixed basis')
paragraph('The observation model is $d=Gh+n$:',r'''The observation model is $d=Gh+n$: $G$ suppresses the signal over one interval, and noise is present throughout the record. We report reconstruction accuracy for $\rho\ge10$ alongside the complete test population, and examine recovery across intervals containing the signal-envelope peak.''','condense repeated evaluation conventions in Introduction')

rep('The SEOBNRv4 remnant prescription implemented in LALSimulation determines',
'The SEOBNRv4 remnant prescription in LALSuite 7.25.1 determines','LALSuite version')
rep('Spherical angular harmonics and luminosity-distance scaling specify the polarizations.',r'''We use spin-weight $-2$ spherical harmonics in the PyCBC convention. The dimensionless strain amplitudes include the mass--distance factor $G_{\rm N}M_f^{\rm det}/(c^2D_L)$, where $G_{\rm N}$ is Newton's constant and $D_L$ the luminosity distance. The complex polarization $h_++ih_\times$ is passed to the response calculation.''','spin-weighted harmonics, mass-distance normalization and complex convention')
rep('PyCBC\'s pre-reference exponential ring-up joins continuously to the damped-mode segment, whose excitation normalization is retained. The envelope has different one-sided slopes at the reference time. This onset convention and the delayed TDI combinations are retained in the reconstruction targets.',r'''For a mode with damping time $\tau_{\ell m}$, the pre-reference amplitude varies as $e^{10t/\tau_{\ell m}}$ for $t_s\le t<0$ and as $e^{-t/\tau_{\ell m}}$ for $t\ge0$. PyCBC sets the common start to $t_s=-\lceil\tau_{\max}/\Delta t\rceil\Delta t$, with $\tau_{\max}$ the longest included damping time; the earlier strain is zero. The oscillatory phase evolves continuously through $t=0$, and both envelope branches have the same amplitude there. The one-sided envelope slopes differ. This prescribed history enters the TDI delays and is included in the reconstruction targets.''','explicit ring-up and truncation')
rep('Reconstruction is evaluated on that finite window, including for signals whose damping times exceed its duration.',
'All reconstruction metrics are computed over the 528-s window, including cases whose dominant-mode damping time exceeds 528 s.','finite-window scope')
rep('The metric uses diagonal Fourier-bin weights. A likelihood for finite records additionally requires the corresponding noise covariance.',
'The metric uses diagonal Fourier-bin weights; a finite-record likelihood requires the corresponding noise covariance.','compact covariance scope')
rep('The first 128 calibration segments determine the expected white-noise RMS, excluding DC and Nyquist, for the coefficient head. All calibration segments are independent of the signal population and test observations.',r'''A separate set of 128 noise realizations calibrates the coefficient head. We whiten these segments, remove DC and Nyquist, average their per-segment mean squares and take the square root in each channel. The resulting noise RMS values are 0.99935 and 0.99909 in $A$ and $E$. Both noise-calibration sets are independent of the source population.''','correct separate 128-noise RMS calibration and aggregation')
rep('Masking is applied to the physical signal before whitening.',
'Signal suppression is imposed numerically on the synthesized TDI $A/E$ series, after the response calculation and before whitening.','specify suppression layer within retained observation model')
paragraph('Gap widths are sampled uniformly',r'''Let $j_p=\arg\max_j\sum_ch_{c,j}^2$ denote the peak sample in the 32768-sample parent response after padding removal. We draw an integer offset $\delta$ uniformly from $-132,\ldots,132$ and set the reference gap start to $g=\operatorname{clip}(j_p+\delta,0,32768-66)$. A second draw selects $a_0$ uniformly from $132,\ldots,263$. The crop origin is $o=\operatorname{clip}(g-a_0,0,32768-N)$, and the local reference start is $a=g-o$. Indices are zero-based, and the reference gap has 66 samples. For a variable gap, $L$ is uniform over the even integers $64,66,\ldots,104$, and its local start is $a_L=a-(L-66)/2$. These widths correspond to 32--52 s and preserve the reference interval center. Fixed 33- and 50-s tests use $L=66$ and $100$. Event-centered cropping assumes prior localization of the event in time; inference receives the cropped observation and mask.''','explicit source-centered gap and crop sampling')

rep('Sixteen scales and a factor-four temporal subsampling give a $2\\times16\\times264$ array per channel.',
'The Morlet parameter is $\\mu=13.4$. We evaluate scales $1,2,\\ldots,16$ using reflection padding and $L^1$ wavelet normalization, and retain every fourth time sample, giving a $2\\times16\\times264$ array per channel.','primary CWT scale, wavelet and boundary conventions')
rep('A second Morlet CWT uses 16 logarithmically spaced frequencies from $0.002$ to $0.25\\,\\mathrm{Hz}$.',
'A second Morlet CWT uses scales $\\mu f_s/(2\\pi f_r)$, with 16 nominal frequencies $f_r$ logarithmically spaced from $0.25$ to $0.002\\,\\mathrm{Hz}$ and the same padding and normalization.','physical-band CWT definition')
rep('computes a db4 decomposition.','computes an eight-level db4 decomposition.','wavelet decomposition depth')
rep('Three convolutional stages with group normalization and GELU map these low-frequency features',
'Both CWTs use the observed channel divided by its RMS, floored at $10^{-8}$. Three convolutional stages with group normalization and GELU map these low-frequency features','CWT observation normalization')

paragraph('DGF architecture. (a)',r'''DGF architecture. The time--frequency and signed temporal representations feed a shared encoder and waveform decoder. Mask-aware and joint-channel refiners precede the signal-basis adapter and conditioned coefficient head. The head adjusts coefficients of a fixed basis; its zero-initialized correction preserves the basis-adapted prediction at initialization. Inverse whitening returns the $A/E$ estimates to physical units. Dashed arrows mark conditioning inputs. The small waveform, transform and network drawings are schematic.}\label{fig:pipeline}\label{fig:architecture}
\end{figure*}''','simplify architecture caption and state initialization condition')
# paragraph replacement includes the figure ending; eliminate the inherited duplicate if present.
s=s.replace('\\end{figure*}\n\\end{figure*}','\\end{figure*}')

rep('The basis and coefficient adapter produce the whitened prediction $b$.',r'''Writing $v$ for the refined waveform, the coefficient adapter produces
\begin{align}
\beta&=B^{\mathsf T}P_0v+0.2s\,g_\varphi(z_v),\nonumber\\
b&=B\beta+(I-P_0)v.\label{eq:basis_adapter}
\end{align}
Here $P_0$ removes DC and Nyquist, $s=\max[\operatorname{RMS}(u),10^{-4}]\sqrt{2N}$, and $g_\varphi$ is a $262\to128\to128\to128$ MLP with GELU after the first two linear layers. The vector $z_v$ contains observation, prediction and mask coordinates defined in Appendix~\ref{app:model_analysis}. Since $P_0B=B$, the interior-frequency component $P_0b$ lies in the span of $B$.''','explicit coefficient adapter and intermediate prediction')
rep('The operator $P_0$ removes DC and Nyquist in each channel. The scale $s=\\operatorname{RMS}(u)\\sqrt{2N}$ uses a small RMS floor for numerical stability. Equation~\\eqref{eq:refinement} preserves the endpoint components of $b$. With a zero-initialized final layer, the head reproduces the starting coefficient prediction to floating-point precision.',
'Equation~\\eqref{eq:refinement} preserves the endpoint components of $b$. At initialization, the final layer gives $F_\\psi=0$ and $c=c_0$. Together with $B^{\\mathsf T}B=I$ and $P_0b\\in\\operatorname{span}(B)$, this yields $\\widehat u=b$ to floating-point precision.','zero-initialization equivalence and its subspace condition')
rep('The 470-dimensional input $z$ contains 262 normalized waveform, coefficient and mask coordinates; 16 absolute-noise and spectral-energy features; and 192 pooled continuous-phase features. The noise features use endpoint-excluded observed RMS, the independently calibrated noise RMS, coarse observed frequency-band energy ratios and RMS residuals between observation and prediction. For the physical-frequency CWT, continuous phase is encoded as',
'The 470-dimensional input $z$ contains 262 normalized coefficient and mask coordinates, 16 noise-level and spectral features, and 192 pooled phase features. Appendix~\\ref{app:model_analysis} defines each component and its normalization. For the physical-frequency CWT, we use an amplitude-weighted circular phase representation,','input definitions and circular phase terminology')
rep('At inference, the coefficient head uses only the observation, current prediction, known mask and fixed noise calibration. Source parameters and clean targets enter simulation, training supervision and evaluation.',
'The head uses the observation, current prediction, mask and noise calibration. Source parameters and clean targets enter simulation, supervision and evaluation.','compact inference inputs')
rep('Each residual block uses layer normalization, two linear layers, GELU and dropout 0.05.',
'Each residual block uses layer normalization, two linear layers, GELU and dropout 0.05, and adds 0.25 times its output to the incoming features.','residual block scale')
rep('The joint-channel mismatch and optimal network signal-to-noise ratio (SNR) are',
'We define the joint-channel mismatch and PSD-weighted network signal-to-noise ratio (SNR) as','PSD-weighted SNR terminology')
rep('We evaluate the overlap at the original time and phase. For nonzero waveforms,',
'The SNR refers to the complete clean target before signal suppression. We evaluate overlap at the original time and phase. For nonzero waveforms,','SNR uses ungapped target and direct overlap convention')
rep('The two-channel training objective is','Final two-channel optimization uses','distinguish final loss from initialization losses')
rep('The objective combines waveform agreement, a noise-weighted amplitude residual and a time-domain residual with three times the weight inside the gap.',
'The objective combines mismatch, normalized noise-weighted waveform error and normalized time-domain waveform error, with a factor-three weight inside the gap.','correct loss interpretation')
paragraph('Figure~\\ref{fig:training_workflow} summarizes the training protocol.',r'''Figure~\ref{fig:training_workflow} summarizes training. The 60,000 physical training sources each contribute three views. Two have independently sampled variable gap widths around the same reference center; the first retains the original noise realization and the second uses independent noise. The third has a 33-s gap and another independent noise realization. The second and third views receive a common input--target sign drawn with equal probability from $\{-1,1\}$ for each source. Widths, noise and signs are drawn once and held fixed across epochs. CWT and db4 features are recomputed from each observation. One epoch contains 180,000 observations.''','exact augmentation relationship and fixed sign draws')
rep('LISA $A/E$ simulation and training. Training, validation and test sources are disjoint. Augmentation uses training sources only; the signal basis is fitted to clean training waveforms, and whitening is calibrated on independent noise. Head selection and staged optimization use the full validation mean. The test observations are generated after the model and protocol are fixed. Dashed arrows indicate validation feedback. Epoch counts are stopping limits, and the small waveform, spectrum and optimization drawings are schematic.',
'Simulation and training workflow. Physical source populations and noise-calibration realizations are separate. Supervised A/E adaptation initializes the waveform estimator and coefficient adapter; head training and joint fine-tuning then use the augmented training views. Validation controls learning rates, early stopping and model selection. The selected estimator is evaluated on newly generated test sources. Dashed arrows show validation feedback; the small drawings are schematic.','simplify workflow caption')
paragraph('The encoder and waveform synthesis modules inherit a pretrained DGF checkpoint;',r'''We initialize the encoder and waveform synthesis modules from a DGF model adapted to the physical $A/E$ training sources. Appendix~\ref{app:initialization} describes the initialization stages and their objectives. Before full coefficient-head training, we compare phase-only, noise-only and combined feature sets on the first 6,000 training sources. All three heads receive the same 262 coefficient and mask coordinates; the phase-only and noise-only variants set the other feature group to zero. The full 1,500-source validation mean selects the combined head. The selected feature set then undergoes two optimization stages. First, the coefficient head is trained with the waveform estimator held fixed, using cached predictions and features. AdamW uses batches of 1024, weight decay $3\times10^{-4}$ and a five-epoch warmup to learning rate $3\times10^{-4}$. Validation plateaus halve the learning rate. Training has a maximum of 600 epochs, minimum 100, patience 100 and improvement threshold $10^{-6}$.''','define warm start and 6000-source feature comparison')

rep('Empirical distributions of time-domain and noise-weighted mismatch for all 6,000 independent test signals. The logarithmic axes show the full range of reconstruction errors. Both metrics use the original signal time and phase.',
'Empirical cumulative distribution functions of time-domain and noise-weighted mismatch for the complete test population. The logarithmic axes show the full range of reconstruction errors.','CDF naming and repeated conventions')
anchor='Figure~\\ref{fig:snr_gap} shows the dependence on SNR and gap location.'
# Current sentence is integrated with SNR range.
rep('Network SNR spans 1.69--923.52, with median 54.77; Fig.~\\ref{fig:snr_gap} shows the dependence on SNR and gap location.',
'Network SNR spans 1.69--923.52, with median 54.77; Fig.~\\ref{fig:snr_gap} shows the dependence on SNR and gap location. A regularized fit in the same training-derived signal basis has mean mismatch 0.04131 for $\\rho\\ge10$, compared with 0.00532 for DGF on identical observations. Appendix~\\ref{app:basis_comparison} specifies this fit and quantifies basis representation error.','add validated signal-basis comparison in Results')
rep('These conditional statistics include the source and noise variations within each group.',
'These statistics reflect the joint distributions of gap location, source parameters and noise realizations within each group.','precise conditional-population interpretation')
gapextra=r'''

We also characterize gap severity by $\eta=\sum_{c,j}m_jh_{c,j}^2/\sum_{c,j}h_{c,j}^2$ and $T_{\rm gap}/\tau_{220}$. In the $\rho\ge10$ population, their medians are 0.289 and 0.238, and their 95th percentiles are 0.666 and 1.44. The 2638 sources with $\eta<0.25$ have mean mismatch 0.00469; the 127 with $\eta\ge0.75$ have mean 0.01583. These conditional summaries include variation in source parameters, noise and gap duration.
'''
rep('Figure~\\ref{fig:examples} shows two $\\rho\\ge10$ sources reconstructed',gapextra+'\nFigure~\\ref{fig:examples} shows two $\\rho\\ge10$ sources reconstructed','gap signal fraction and damping-time coverage')

rep('All four curves use the same clean-signal amplitude normalization per source and connect the original 0.5-s samples. Source seeds, from top to bottom, are 2038202231 and 2038201952. Full-population statistics use all 6,000 test sources.',
'For each source, all curves in both channels are divided by $H_{\\rm TDI}=\\max_{c,j}|h_{c,j}|$ and connect the original 0.5-s samples. Source seeds, from top to bottom, are 2038202231 and 2038201952. The example criteria are given in Appendix~\\ref{app:model_analysis}.','waveform normalization formula and neutral example-criteria reference')
rep('The common unwrapped phase branch preserves the relative phase modulo $2\\pi$; the arrival time and phase remain fixed.',
'The clean phase is $\\phi_c=\\operatorname{unwrap}(\\arg z_c)$; the displayed reconstruction is $\\phi_c+\\arg(\\widehat z_cz_c^*)$. Unwrapping removes adjacent jumps greater than $\\pi$.','actual phase-display algorithm')
rep('Across all 6,000 test sources, the mean absolute phase errors are',
'For the complete test population, the mean absolute phase errors are','remove repeated sample count')
rep('Phase evolution through the resolved oscillations is shown in Fig.~\\ref{fig:phase}(b).',
'Figure~\\ref{fig:phase}(b) compares analytic phases for the $A$ channel of one example.','single-source phase illustration scope')
phase_extra=r'''

To measure recovery within the interval, we define the envelope-weighted gap phase error
\begin{equation}
\Phi_{\rm gap}=\left[\frac{\sum_{c,j}m_j|z_{c,j}|^2[\arg(\widehat z_{c,j}z_{c,j}^*)]^2}{\sum_{c,j}m_j|z_{c,j}|^2}\right]^{1/2}.\label{eq:gap_phase}
\end{equation}
Among the 1885 sources with $\rho\ge10$ and the joint envelope peak inside the gap, its mean, median and 95th percentile are $4.92^\circ$, $3.89^\circ$ and $11.61^\circ$. Over the complete peak-inside-gap population of 1935 sources, the corresponding mean is $5.51^\circ$.
'''
rep('Individual quasinormal-mode phases require a separate mode decomposition.',
'Individual quasinormal-mode phases require a separate mode decomposition.'+phase_extra,'add gap-interior phase diagnostic')

rep('Across all 6,000 test sources, $E$ has mean',
'For the complete test population, $E$ has mean','amplitude sample scope without repeated count')
rep('The norm bias measures overall amplitude scaling, while $E$ also includes errors in waveform shape.',
'The norm bias measures the fractional difference between reconstructed and target waveform norms; $E$ measures their full waveform difference.','norm bias interpretation')
rep('The denominator measures the clean signal contained within the interval.',
'The denominator is the Euclidean norm of the clean waveform restricted to the gap.','gap-local denominator definition')
paragraph('Figures~\\ref{fig:tdi_log_amplitudes} and~\\ref{fig:psd} show',r'''Figure~\ref{fig:tdi_log_amplitudes} plots waveform absolute values for the sources in Fig.~\ref{fig:examples}. In these two examples, the logarithmic scale reveals differences in the late-time waveform and near zero crossings. Figure~\ref{fig:psd} compares population-averaged target, residual and instrument-noise spectra. For $\rho\ge10$, the mean normalized noise-weighted squared waveform error $E_S$ is 0.01190; its complete-population mean is 0.01786. The periodogram definition is given in Appendix~\ref{app:model_analysis}.''','remove unmeasured tail-growth claim; add target PSD and integrated error')
rep('Absolute amplitudes of the clean and reconstructed $A$ (left) and $E$ (right) waveforms',
'Absolute values of the clean and reconstructed $A$ (left) and $E$ (right) waveforms','absolute values versus envelope')
rep('All curves use the same physical-time arrays as Fig.~\\ref{fig:examples}, with native sampling and fixed arrival time and phase.',
'The curves connect the native samples used in Fig.~\\ref{fig:examples}.','remove repeated plotting convention')
rep('Mean residual and noise power spectral densities in the $A/E$ channels over all 6,000 test sources. The reconstruction residual is $\\widehat h-h$. Fractional-frequency amplitudes are dimensionless, so the PSD has units $\\mathrm{Hz}^{-1}$. The curves use the same one-sided periodogram convention.',
'Mean power spectral densities of the target signal $h$, reconstruction residual $\\widehat h-h$ and instrument noise in the $A/E$ channels over all test sources. Each periodogram uses a rectangular window, constant detrending and one-sided density normalization, followed by an arithmetic mean over sources. Only interior Fourier bins are shown. Fractional-frequency PSDs have units $\\mathrm{Hz}^{-1}$.','PSD target reference and exact computation convention')
rep('A shared encoder and joint-channel refinement recover the clean LISA $A/E$ waveforms from the noisy observation and gap mask.',
'A shared encoder and joint-channel refinement estimate the clean LISA $A/E$ waveforms from the noisy observation and gap mask.','estimate rather than universal recovery')
rep('Weak late-time oscillations contribute less to the full-record metrics and retain larger relative discrepancies, as the logarithmic amplitude plots illustrate.',
'The two waveform examples show visible late-time differences on a logarithmic scale. Their contribution to a full-record error depends on its weighting.','limit late-time interpretation to measured examples')
rep('The fixed-duration tests and reference comparison in Appendix~\\ref{app:fixed_gaps} complement the variable-gap results; comparisons with other estimators on identical observations would establish their relative reconstruction accuracy.',
'The fixed-duration tests and signal-basis comparison complement the variable-gap results. The latter separates linear fitting in the learned subspace from DGF coefficient prediction.','interpret added comparison without untested superiority claim')
rep('Moving unequal-arm constellations, confusion foregrounds, nonstationary noise, overlapping sources and multiple gaps require simulations spanning those conditions.',
'Tests with moving unequal-arm constellations and nonstationary noise would assess sensitivity to response and noise-model errors. Independently perturbed mode amplitudes, phases and frequencies would test how much structure outside the training relations is preserved.','specific future tests and waveform-prior scope')
rep('Parameter inference uses the original observation likelihood, constructed from $d-Gh(\\theta)$ and the noise covariance. Posterior bias and coverage tests would establish the accuracy of that inference, including for gaps near the signal peak.',
'A subsequent parameter analysis under Eq.~\\eqref{eq:observation} would use a likelihood constructed from $d-Gh(\\theta)$ and the noise covariance. Posterior-bias and coverage tests remain part of that analysis.','future inference status and observation-model scope')

rep('The stored parameter count includes unused multihead-attention tensors allocated by the implementation. ',
'','move implementation-only parameter details to companion record')
rep('The db4 representation uses the approximation and seven detail components in library coefficient order.',
'The eight db4 channels reconstruct $A_8,D_8,D_7,\\ldots,D_2$ with zero padding.','explicit db4 components and boundary convention')
rep('Its linear patch map and the encoder embedding map have independent checkpoint values. When loading, we clone the initially transposed storage before reading the tensors so that their saved values are preserved on both CPU and GPU.',
'The encoder embedding and decoder patch maps use independent parameter tensors.','remove storage-loading engineering log')
paragraph('The $A/E$ test seeds are',r'''Source seeds are consecutive integers starting at 2028000000, 2028100000 and 2038200000 for training, validation and testing, respectively. The selected checkpoint has SHA-256 prefix \texttt{0ffce08b8856}; the reproducibility manifest gives full hashes and separate noise seeds.''','compact identifiable experiment record')
paragraph('The simulated response peak determines the event-centered crop',r'''The implementation uses PyCBC 2.8.2, LALSuite 7.25.1, \texttt{fastlisaresponse} 1.1.17, LISA Analysis Tools 1.2.8, \texttt{ssqueezepy} 0.6.5 and \texttt{pytorch\_wavelets} 1.3.0. The computational source version is \texttt{ae47cce8a5c0}. Code is maintained at \url{https://github.com/ljq2088/RingdownGapFilling}. Checkpoints, calibration arrays and evaluation records are available from the authors on request; the accompanying reproducibility manifest identifies the assets and software versions.''','software versions and artifact availability')
paragraph('The two sources in Fig.~\\ref{fig:examples} have envelope peaks',r'''Figure~\ref{fig:examples} uses sources with the joint peak in the gap, $\rho\ge10$, at least four $\tau_{220}$ of post-peak coverage and six visible extrema per channel. Full-record and gap-local errors are below 6\%, and peak-phase errors below $2.5^\circ$. The plotting manifest gives the channel-balance, native-sample curvature and noise-amplitude criteria.''','neutral and explicit waveform example rule')
paragraph('The phase statistics in Sec.~\\ref{subsec:PhaseRecoveryAlignment}',r'''Each channel-specific phase distribution contains one value per test source. Gap-local errors use the target norm within the gap; the peak-inside-gap population provides intervals containing appreciable signal.''','compact phase and gap-local conventions')

# Add precise feature definitions, spectral convention and initialization procedure.
appendix=r'''

\subsection{Coefficient-head inputs}
For an intermediate prediction $y\in\{v,b\}$, let $a_y=B^{\mathsf T}P_0y$, $q=B^{\mathsf T}u$, $r=\operatorname{RMS}(u)$ over both channels and $s=\max(r,10^{-4})\sqrt{2N}$. Write $R_c(x)=\operatorname{RMS}(P_0x_c)$ and $\nu_c$ for the calibrated whitened-noise RMS. The coefficient adapter uses the 262-dimensional coordinates for $y=v$; the final head uses those for $y=b$ and adds noise and phase features. Table~\ref{tab:head_features} lists the components. Before computing noise features, $R_c(u)$ and $\nu_c$ are floored at $10^{-6}$. Log ratios are clipped to $[-8,8]$ and logarithms of the form $\log(1+x)$ to a maximum of 8.

\begin{table*}[t]\centering
\caption{Input features of the conditioned coefficient head. The first group also supplies the coefficient adapter. $a$ and $L$ denote the local gap start and length. The phase features use the physical-frequency CWT.}\label{tab:head_features}
\begin{tabular}{lrl}\hline\hline
Group & Dimension & Components\\\hline
Prediction coefficients & 128 & $a_b/s$\\
Observation coefficients & 128 & $q/s$\\
Gap coordinates & 3 & $a/N$, $(a+L)/N$, $L/N$\\
Channel RMS ratios & 2 & $\operatorname{RMS}(u_c)/\max(r,10^{-4})$\\
Coefficient norm & 1 & $\|a_b\|_2/s$\\\hline
Noise-level ratios & 4 & $\log[R_c(u)/\nu_c]$, $\log[1+\nu_c/R_c(u)]$, for each channel\\
Frequency-band ratios & 8 & $[2\sum_{k\in K}|\operatorname{rfft}(u_c)_k|^2/N^2]^{1/2}/R_c(u)$\\
Residual and predicted RMS & 4 & $\log[1+R_c(u-b)/\nu_c]$, $\log[1+R_c(b)/\nu_c]$\\\hline
Pooled phase representation & 192 & $\log(1+|C|)\{1,\cos\theta_C,\sin\theta_C\}$; $4\times8$ pooling per channel\\
\hline\hline\end{tabular}

\smallskip\parbox{0.96\textwidth}{The four frequency-bin groups are $K=\{1,\ldots,4\}$, $\{5,\ldots,16\}$, $\{17,\ldots,64\}$ and $\{65,\ldots,527\}$. Both channels contribute to each feature group, giving $262+16+192=470$ inputs.}
\end{table*}

\subsection{Periodograms}
For a series $x_j$, the one-sided periodogram used in Fig.~\ref{fig:psd} is
\begin{equation}
P_x(f_k)=\frac{\kappa_k}{Nf_s}\left|\sum_{j=0}^{N-1}(x_j-\bar x)e^{-2\pi i jk/N}\right|^2,
\end{equation}
with $\kappa_k=2$ at interior bins and 1 at DC and Nyquist. We use a rectangular window and average the individual source periodograms arithmetically. The target, residual and noise series receive identical processing.

\subsection{Initialization and adaptation}\label{app:initialization}
The supplied legacy DGF weights initialize reproduction. Their training code uses pointwise squared and absolute errors on a separate 10,000-waveform simulation bank. Physical $A/E$ adaptation uses 12,000 training sources with 33-s gaps and objective $\mathcal M_{AE}+0.5E_S+0.5E_{\rm time,gap}$, then 60,000 sources for joint-channel and physical-frequency adaptation with an additional Hann-windowed penalty.

The signal basis uses the 60,000 clean training directions. Coefficient-adapter training and variable-gap joint training use $\mathcal M_{AE}+0.5E_S+0.05E_{\rm time,gap}+0.02[\mathcal M_H+0.5(\sqrt{1+E_H}-1)]$, where $\mathcal M_H,E_H$ use a Hann window and its noise calibration. All physical-data stages select on the same complete validation mean. Their configurations and weights are identified in the manifest. The final-head initialization has identifier \texttt{a1e9959a7a95}; final optimization uses Eq.~\eqref{eq:loss} and Sec.~\ref{sec:training}, followed by new independent testing.
'''
rep('\\section{Fixed-duration gaps and reference comparison}',appendix+'\n\\section{Fixed-duration gaps and reference comparison}','feature table, periodogram definition and initialization stages')
rep('It uses the preceding fitted synthesis parameters and omits the final noise-conditioned head.',
'This reference uses the variable-gap joint-training checkpoint \\texttt{a1e9959a7a95}, comprising the waveform estimator and coefficient adapter.','identify reference configuration')
rep('The paired mean difference is $-1.61\\times10^{-4}$, with 95\\% bootstrap interval $[-2.45\\times10^{-4},-8.11\\times10^{-5}]$. The interval uses 5,000 paired resamples of the independent test sources.',
'For each source, define $\\Delta\\mathcal M=\\mathcal M_{\\rm DGF}-\\mathcal M_{\\rm ref}$. Its mean is $-1.61\\times10^{-4}$. A paired bootstrap with 5,000 source-level resamples gives the 95\\% percentile interval $[-2.45\\times10^{-4},-8.11\\times10^{-5}]$ for the mean difference.','paired difference direction and percentile bootstrap')

basis=json.loads((P/'basis_prior_corrected/basis_results.json').read_text())
diagnostic=json.loads((P/'additional_diagnostics.json').read_text())
extra=r'''

\section{Signal-basis fitting and representation error}\label{app:basis_comparison}
We compare DGF with a linear estimator using the same rank-128 basis $B$. Four orthonormal DC and Nyquist directions, collected in $B_{\rm end}$, supply endpoint degrees of freedom. Each clean training waveform is divided by $\|P_0Wh\|_2$ before projecting onto $[B,B_{\rm end}]$. The mean squared coefficients define prior variances, floored at $10^{-16}$. Let $D$ contain their square roots and $Q=[B,B_{\rm end}]D$. The regularized estimate is
\begin{align}
\widehat a_\lambda&=\arg\min_a\{\|P_0W(d-GW^{-1}Qa)\|_2^2+\lambda\|a\|_2^2\},\nonumber\\
\widehat h_{\rm lin}&=W^{-1}Q\widehat a_\lambda.
\end{align}
Thus the forward map applies the same physical signal suppression before whitening. The Gaussian coefficient prior uses training data only. A grid containing zero and $10^{-6},10^{-5},\ldots,10^2$ selects $\lambda=10^{-3}$ by the complete validation mean. Table~\ref{tab:basis_fit} compares this estimator and DGF on identical test observations. The complete-population paired mean difference, DGF minus linear fit, is $-0.04341$, with 95\% percentile bootstrap interval $[-0.04532,-0.04156]$ from 5,000 source-level resamples.

\begin{table}[t]\centering\setlength{\tabcolsep}{3pt}
\caption{Direct noise-weighted mismatch for DGF and the training-prior regularized basis fit.}\label{tab:basis_fit}
\begin{tabular}{llrrr}\hline\hline
Population & Estimator & Mean & Median & 95th perc.\\\hline
$\rho\ge10$ & DGF & 0.00532 & 0.00241 & 0.01970\\
& Basis fit & 0.04131 & 0.02030 & 0.14912\\
All sources & DGF & 0.00852 & 0.00254 & 0.02782\\
& Basis fit & 0.05193 & 0.02141 & 0.21892\\
\hline\hline\end{tabular}\end{table}

To isolate representation error, we project each clean target onto the first $r$ columns of $B$ in whitened coordinates. For $r=32,64,128$, the mean mismatches are 0.03690, 0.00410 and $1.65\times10^{-4}$ over all sources; the corresponding $\rho\ge10$ means are 0.03322, 0.00319 and $1.11\times10^{-4}$. This target projection measures the subspace approximation error.

'''
rep('\\clearpage\n\\bibliography{DGF_references}',extra+'\n\\clearpage\n\\bibliography{DGF_references}','validated basis benchmark, rank representation error and gap-severity statistics')

# Remove a redundant statement already defined in Methods.
rep('All sources contribute to the complete-population statistics.','The two groups partition the test population.','gap-location figure caption')

# Keep the two original numeric tables and core metric/loss definitions intact.
for label in ('eq:joint_inner','eq:joint_mismatch','eq:loss','eq:auxiliary_errors','eq:whitening'):
    def env(text):
        end=text.index('\\label{'+label+'}')
        begin=max(text.rfind('\\begin{equation}',0,end),text.rfind('\\begin{align}',0,end))
        stop=min(x for x in (text.find('\\end{equation}',end),text.find('\\end{align}',end)) if x>=0)
        return text[begin:stop]
    assert env(original)==env(s),label
assert re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}',original,re.S)==re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}',s,re.S)[:1]+[re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}',s,re.S)[2]]
assert len(re.findall(r'\\includegraphics',s))==8
assert all(w.lower() not in s.lower() for w in ('Selected successful','four-component','0.109','Tail-relative residuals grow','optimal network','Atomic model','white-noise RMS'))

# Preserve the reference bank, changing active software records and formatting.
bib=oldbib
def entry(key):
    start=re.search(r'@\w+\s*\{\s*'+re.escape(key)+r'\s*,',bib).start();i=bib.index('{',start);n=1;j=i+1
    while n:
        if bib[j]=='{' and bib[j-1]!='\\':n+=1
        elif bib[j]=='}' and bib[j-1]!='\\':n-=1
        j+=1
    return bib[start:j]
def edit(key,fn):
    global bib
    old=entry(key);new=fn(old);bib=bib.replace(old,new,1)
edit('PyCBCRingdown',lambda e:e.replace('time-domain ringdown waveform documentation','2.8.2 ringdown waveform implementation').replace('https://pycbc.org/pycbc/latest/html/pycbc.waveform.html','https://github.com/gwastro/pycbc/blob/v2.8.2/pycbc/waveform/ringdown.py'))
edit('LISAAnalysisTools',lambda e:e.replace('detector and sensitivity implementation','version 1.2.8: detector and sensitivity implementation').replace('https://github.com/lisa-analysis-tools/lisa-analysis-tools}','https://github.com/lisa-analysis-tools/lisa-analysis-tools/releases/tag/v1.2.8}'))
substitutions={'Physical Review D':'Phys. Rev. D','Physical Review Letters':'Phys. Rev. Lett.','Classical and Quantum Gravity':'Class. Quantum Grav.','Monthly Notices of the Royal Astronomical Society':'Mon. Not. R. Astron. Soc.','Physics Letters B':'Phys. Lett. B','Machine Learning: Science and Technology':'Mach. Learn.: Sci. Technol.','International Journal of Modern Physics A':'Int. J. Mod. Phys. A','IEEE Transactions on Medical Imaging':'IEEE Trans. Med. Imaging','IEEE Signal Processing Magazine':'IEEE Signal Process. Mag.','Proceedings of the IEEE':'Proc. IEEE','IEEE Transactions on Acoustics, Speech, and Signal Processing':'IEEE Trans. Acoust. Speech Signal Process.','Transactions on Machine Learning Research':'Trans. Mach. Learn. Res.'}
active=set(k.strip() for group in re.findall(r'\\cite\{([^}]+)\}',s) for k in group.split(','))
for key in active:
    def format_entry(e):
        for a,b in substitutions.items():e=e.replace('{'+a+'}','{'+b+'}')
        # Preserve proper nouns under the APS bibliography title conversion.
        for a,b in [('discrete Fourier','discrete {Fourier}'),('Discrete Fourier','Discrete {Fourier}'),('A Bayesian','A {Bayesian}'),('Galactic-binary','{Galactic}-binary')]:e=e.replace(a,b)
        return e
    edit(key,format_entry)

(P/'source/DGF.tex').write_text(s,encoding='utf-8',newline='\n')
(P/'source/DGF_references.bib').write_text(bib,encoding='utf-8',newline='\n')
for name,a,b in [('manuscript',original,s),('bibliography',oldbib,bib)]:
    (P/(name+'.patch')).write_text(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='before/'+name,tofile='after/'+name)),encoding='utf-8')
(P/'editorial_changes.json').write_text(json.dumps(dict(changes=changes,count=len(changes),original_tables_preserved=True,core_metric_and_loss_unchanged=True,selected_model_and_test_population_unchanged=True),indent=2),encoding='utf-8')
print('Revised',len(changes),'passages and added validated diagnostics.')
