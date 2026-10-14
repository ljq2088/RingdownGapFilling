from pathlib import Path
import re, json, hashlib, difflib

HERE=Path(__file__).resolve().parent
P=HERE/'source/DGF.tex'
original=P.read_text(encoding='utf-8')
s=original
changes=[]

def replace(old,new,tag):
    global s
    assert s.count(old)==1,(tag,s.count(old))
    s=s.replace(old,new)
    changes.append(tag)

start=s.index('Data gaps can obscure')
end=s.index(r'\end{abstract}',start)
replace(s[start:end],r'''Data gaps near the onset of black hole ringdown can obscure much of its rapidly decaying signal. We present DenoiseGapFiller (DGF), a neural network that jointly denoises and reconstructs ringdown waveforms using complementary time--frequency and temporal representations. We evaluate DGF on 6,000 independent simulated LISA observations with gaps spanning approximately 6\%--10\% of each record. The mean noise-weighted mismatch, evaluated at fixed arrival time and phase, is $8.52\times10^{-3}$ across the test population and $5.32\times10^{-3}$ for network signal-to-noise ratios $\rho\ge10$. Mean absolute phase errors at the signal-envelope peak are below $4^\circ$ in both $A/E$ channels. These results demonstrate recovery of the oscillatory waveform across gaps, including intervals containing the signal peak, within the simulated source population and stationary Gaussian instrument-noise model.
''','abstract')

replace('An interval covering 20\\% of a short record illustrates a severe interruption; the variable-duration LISA protocol considered here covers approximately 6\\%--10\\% of a 528-s record.',
'Gaps covering as much as 20\\% of a short ringdown record illustrate the severity of this problem. Here we study gaps spanning approximately 6\\%--10\\% of a 528-s record.','background gap scope')

replace('Sparse inpainting exploits structure in Fourier or wavelet representations and has been applied to LISA galactic-binary signals~\\cite{Blelly2021,Starck2010}.',
'Sparse representations provide a basis for inpainting~\\cite{Starck2010}; Fourier sparsity has been used to recover Galactic-binary signals in gapped LISA data~\\cite{Blelly2021}.','sparse literature scope')

replace('These studies motivate learned waveform estimation. Their waveform families, detector responses and noise conditions also define the domains in which their reported performance applies.',
'Here we focus on short ringdown records, where a gap can cover the amplitude peak and leave substantially different signal strengths on its two sides.','concrete motivation')

replace('The time--frequency branch represents local oscillatory content through complex Morlet coefficients.',
'The time--frequency branch represents local oscillatory content through complex Morlet continuous wavelet transform (CWT) coefficients.','CWT definition')
replace('The LISA configuration incorporates PSD whitening, mask-aware and joint-channel modules, and noise-conditioned coefficients in a signal basis fitted to clean training waveforms.',
'Whitening uses a calibrated power spectral density (PSD), while mask-aware and joint-channel modules refine the decoded waveform. A coefficient head conditions a training-derived signal basis on the observed noise level and phase features.','plain architecture prose')

replace('The observation model is $d=Gh+n$: $G$ suppresses the signal over one interval, and noise is present throughout the record. We assess recovery through direct waveform mismatch, amplitude residuals, error spectra and per-channel analytic phase. Full-population distributions establish the typical accuracy and its upper tail; separate peak-masking statistics resolve the effect of losing the strongest part of the signal. Waveform comparisons show the recovered oscillations across gaps containing the signal-envelope peak. All reconstruction results use one selected DGF model and the independent LISA test population defined in Sec.~\\ref{sec:method}. The source and channel conventions are common to training, validation and testing.',
'The observation model is $d=Gh+n$: $G$ suppresses the signal over one interval, and noise is present throughout the record. We evaluate waveform mismatch at fixed arrival time and phase, together with amplitude residuals, error spectra and per-channel analytic phase. Population distributions quantify accuracy across the full test set, and conditional statistics distinguish gaps containing the signal-envelope peak.','evaluation framing')

replace('The appendixes provide implementation details, additional waveform diagnostics and tests with fixed gap durations.',
'The appendixes give implementation and reproducibility details, fixed-duration gap tests and a comparison with a reference DGF configuration.','roadmap')

replace('DGF operates on the $A/E$ TDI channels and estimates their complete clean waveforms from the observations and known gap mask.',
'DGF operates on the $A/E$ time-delay interferometry (TDI) channels and estimates their complete clean waveforms from the observations and known gap mask.','TDI definition')
replace('The metric weights Fourier bins diagonally; likelihood calculations must also account for correlations introduced by the finite window.',
'The metric uses diagonal Fourier-bin weights. A likelihood for finite records additionally requires the corresponding noise covariance.','finite-window covariance')
replace('The parent response peak sets the event-centered simulation window, with reference gap offsets drawn between $-66$ and $66$ s.',
'The parent response peak sets the event-centered simulation window, with reference gap offsets drawn between $-66$ and $66$ s. This setup assumes that the event has already been localized in time.','event timing scope')
replace('The head maps 470 inputs to width 192, applies two residual MLP blocks and outputs 128 coefficients.',
'The head maps 470 inputs to width 192, applies two residual multilayer perceptron (MLP) blocks and outputs 128 coefficients.','MLP definition')
replace(r'\subsection{Direct mismatch, training objective and source separation}',
r'\subsection{Training objective and data partition}','subsection terminology')
replace('The model is initialized from a pretrained DGF reconstruction model, with a zero-initialized output layer in the noise-conditioned coefficient head. Training proceeds in two stages.',
'The encoder and waveform synthesis modules inherit a pretrained DGF checkpoint; the new coefficient head has a zero-initialized output layer. The initialization checkpoint and fixed preprocessing assets are identified by SHA-256 hashes in the archived training record. Optimization of this estimator proceeds in two stages.','initialization clarity')

replace('All results in this section and the appendixes use the same selected DGF model and 6,000 independent LISA test sources. Training and validation sources are disjoint from this test population. Each record contains 1056 samples at 2 Hz, corresponding to 528 s. The primary test assigns one gap of 64--104 samples, sampled uniformly from the even lengths in this range, giving durations of 32--52 s. The model is evaluated once after validation selection, and every source contributes to the reported population statistics.',
'We evaluate the validation-selected checkpoint on 6,000 independent LISA test sources. Each 528-s record contains 1056 samples at 2 Hz and one 32--52-s gap drawn as specified in Sec.~\\ref{sec:gaps}. Every source contributes to the full-population statistics. The fixed-duration tests in Appendix~\\ref{app:fixed_gaps} use the same source waveforms and noise realizations.','test protocol precision')

replace('The mean noise-weighted mismatch is 0.00852, with median 0.00254 and 95th percentile 0.02782. The mean time-domain mismatch is 0.01367. Figure~\\ref{fig:mismatch} gives the complete distributions. Network SNR, defined by Eq.~\\eqref{eq:joint_mismatch}, ranges from 1.69 to 923.52, with median 54.77. The $\\rho\\ge10$ subset contains 5814 sources and has mean noise-weighted mismatch 0.00532. Figure~\\ref{fig:snr_gap} resolves the dependence on SNR and gap location.',
'Table~\\ref{tab:results} and Fig.~\\ref{fig:mismatch} summarize the reconstruction accuracy. The median noise-weighted mismatch is 0.00254, while the mean is 0.00852 and the 95th percentile is 0.02782, reflecting an upper tail of larger errors. Network SNR ranges from 1.69 to 923.52, with median 54.77. The $\\rho\\ge10$ subset contains 5814 sources and has mean noise-weighted mismatch 0.00532. Figure~\\ref{fig:snr_gap} shows how the error varies with SNR and gap location.','results interpretation')

replace('Both subsets retain variation in source parameters and noise realizations.',
'These conditional populations also differ in source parameters and noise realizations, so their comparison describes the test population with each gap location.','conditional interpretation')
replace('The windows extend for approximately 4.9 dominant-mode damping times after the response peak.',
'For these two sources, the windows extend for approximately 4.9 dominant-mode damping times after the response peak.','examples finite-window scope')

replace('This diagnostic evaluates the analytic phase of the channel waveform at fixed time; quasinormal-mode phase parameters require a separate mode decomposition.',
'These statistics describe the analytic phase of each channel at its clean-signal envelope peak. Phase evolution through the resolved oscillations is shown in Fig.~\\ref{fig:phase}(b). Individual quasinormal-mode phases require a separate mode decomposition.','phase diagnostic scope')

replace('Together with mismatch, these measurements characterize waveform shape and overall amplitude fidelity.',
'The norm bias measures overall amplitude scaling, while $E$ also includes errors in waveform shape.','amplitude interpretation')
replace('The amplitude panels use the sources in Fig.~\\ref{fig:examples} and resolve the decaying tail and the waveform zero crossings.',
'The amplitude panels use the sources in Fig.~\\ref{fig:examples}. The logarithmic scale reveals differences in the weak tail and near zero crossings that are less visible in the signed waveform plots. Tail-relative residuals grow as the clean envelope decays; the full-record overlap and the peak-phase statistic emphasize different parts of this structure.','tail visibility')

a=s.index('We have presented DGF for joint denoising',s.index(r'\section{Discussion'))
b=s.index(r'\begin{acknowledgments}',a)
replace(s[a:b],r'''DGF reconstructs black hole ringdown waveforms by combining time--frequency information with signed multiscale temporal features. A shared encoder and joint-channel refinement recover the clean LISA $A/E$ waveforms from the noisy observation and gap mask. On 6,000 independent test sources with 32--52-s gaps, the mean direct noise-weighted mismatch is 0.00852, decreasing to 0.00532 for $\rho\ge10$. The waveform comparisons demonstrate recovery across intervals containing the signal peak, and the population residuals quantify the remaining amplitude error.

The mean absolute analytic-phase errors at the envelope peak are $3.84^\circ$ and $3.79^\circ$ in $A$ and $E$. Weak late-time oscillations contribute less to the full-record metrics and retain larger relative discrepancies, as the logarithmic amplitude plots illustrate. Assessing individual mode frequencies, damping times and phases requires mode-resolved fits over specified time intervals. The fixed-duration tests and reference comparison in Appendix~\ref{app:fixed_gaps} complement the variable-gap results; comparisons with other estimators on identical observations would establish their relative reconstruction accuracy.

The simulations cover nonspinning progenitor binaries, an idealized equal-arm constellation and stationary Gaussian instrument noise. They assume an event-centered window, one known gap and the signal history specified in Sec.~\ref{sec:physical_signals}. Tests with continuous merger histories would assess sensitivity to the pre-reference ring-up entering the TDI delays. Moving unequal-arm constellations, confusion foregrounds, nonstationary noise, overlapping sources and multiple gaps require simulations spanning those conditions.

DGF provides a clean-waveform point estimate for inspecting ringdown structure and initializing waveform fits. Parameter inference uses the original observation likelihood, constructed from $d-Gh(\theta)$ and the noise covariance. Posterior bias and coverage tests would establish the accuracy of that inference, including for gaps near the signal peak. Previous LISA gap studies~\cite{Baghi2019,Mao:2025cae} motivate this validation alongside waveform-level metrics. Quantifying reconstruction uncertainty and testing the recovery of individual modes are the next steps toward applications in black hole spectroscopy.

''','discussion')

replace('The white-space RMS floor is $10^{-4}$ for coefficient and residual normalization.',
'The RMS floor in whitened coordinates is $10^{-4}$ for coefficient and residual normalization.','whitened coordinates')
replace('The implementation also allocates multihead-attention parameters that are inactive in the forward pass. The quoted total parameter count includes those stored tensors.',
'The stored parameter count includes unused multihead-attention tensors allocated by the implementation.','attention implementation precision')
replace('The figure data include source seeds, waveform parameters, per-source diagnostics and native-sample curvature used to construct the comparison.',
'The archived figure arrays contain the source seeds, waveform parameters and per-source diagnostics.','remove plotting diary')

# Preserve every equation and numerical table, and preserve the identity and ordering of figures.
for env in ('equation','align','tabular'):
    pat=r'\\begin\{'+env+r'\}.*?\\end\{'+env+r'\}'
    assert re.findall(pat,original,re.S)==re.findall(pat,s,re.S),env
assert re.findall(r'\\includegraphics[^\n]+',original)==re.findall(r'\\includegraphics[^\n]+',s)
P.write_text(s,encoding='utf-8',newline='\n')

bp=HERE/'source/DGF_references.bib'
bo=bp.read_text(encoding='utf-8'); bib=bo
def entry(key):
    a=re.search(r'@\w+\s*\{\s*'+re.escape(key)+r'\s*,',bib).start()
    i=bib.index('{',a);n=1;j=i+1
    while n:
        if bib[j]=='{' and bib[j-1]!='\\':n+=1
        elif bib[j]=='}' and bib[j-1]!='\\':n-=1
        j+=1
    return bib[a:j]
def changeentry(key,transform):
    global bib
    old=entry(key); new=transform(old);assert old!=new,key
    bib=bib.replace(old,new,1)

changeentry('dey2021gaps',lambda e:e.replace('pages={043004}','pages={044035}').replace('with LISA','with {LISA}').replace('publisher={APS}','publisher={APS},\n  doi={10.1103/PhysRevD.104.044035},\n  eprint={2104.12646}, archivePrefix={arXiv}'))
changeentry('Blelly2021',lambda e:e.replace('journal = {arXiv preprint}','journal = {Monthly Notices of the Royal Astronomical Society},\n  volume = {509}, number = {4}, pages = {5902--5917},\n  doi = {10.1093/mnras/stab3314}, archivePrefix = {arXiv}').replace('year    = {2021}','year    = {2022}'))
changeentry('Isi_2019',lambda e:e.replace('with GW150914','with {GW150914}').replace('10.1103/physrevlett.123.111102','10.1103/PhysRevLett.123.111102').replace('number={11}','pages={111102},\n   number={11}'))
changeentry('TSMixer2023',lambda e:e.replace('Li, Chun-Liang and Arik','Li, Chun-Liang and Yoder, Nate and Arik'))
changeentry('Horowitz1974',lambda e:r'''@article{Horowitz1974,
  author={Horowitz, L. L.},
  title={The effects of spline interpolation on power spectral density},
  journal={IEEE Transactions on Acoustics, Speech, and Signal Processing},
  volume={22}, number={1}, pages={22--27}, year={1974},
  doi={10.1109/TASSP.1974.1162536}
}''')
for key,doi in [('berti2006gravitational','10.1103/PhysRevD.73.064030'),('berti2009qnm','10.1088/0264-9381/26/16/163001'),('robson2019construction','10.1088/1361-6382/ab1101')]:
    changeentry(key,lambda e,d=doi:e[:-1].rstrip()+',\n  doi={'+d+'}\n}')
for key in ('berti2006gravitational','robson2019construction','Baghi2019'):
    changeentry(key,lambda e:e.replace('LISA','{LISA}'))
changeentry('Luo2016',lambda e:e.replace('TianQin:','{TianQin}:').replace('Luo, J. and others (TianQin Collaboration)','Luo, Jun and others'))
changeentry('Ruan2020',lambda e:e.replace('Taiji program','{Taiji} program'))
changeentry('Wang:2024waveformer',lambda e:e.replace('title = {WaveFormer:','title = {{WaveFormer}:'))
bp.write_text(bib,encoding='utf-8',newline='\n')

(HERE/'manuscript.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='before/DGF.tex',tofile='after/DGF.tex')),encoding='utf-8')
(HERE/'bibliography.patch').write_text(''.join(difflib.unified_diff(bo.splitlines(True),bib.splitlines(True),fromfile='before/DGF_references.bib',tofile='after/DGF_references.bib')),encoding='utf-8')
summary=dict(changes=changes,paragraph_groups=len(changes),equations_and_tables_unchanged=True,figure_includes_unchanged=True,word_count_before=len(re.findall(r'\b[\w-]+\b',original)),word_count_after=len(re.findall(r'\b[\w-]+\b',s)),tex_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),bib_sha256=hashlib.sha256(bp.read_bytes()).hexdigest())
(HERE/'editorial_changes.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
