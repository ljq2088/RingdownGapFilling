from pathlib import Path
import json, zipfile, re, hashlib, difflib
P=Path(__file__).resolve().parent; O=P/'source'
R=Path(r'\\wsl.localhost\Ubuntu-22.04-D\home\ljq\code\RingdownGapFilling_PRD'); F=R/'FillingGap/results_prd_tail_20261015'
with zipfile.ZipFile(P/'overleaf_before.zip') as z:
    t=z.read('DGF.tex').decode('utf8'); bib=z.read('DGF_references.bib').decode('utf8'); manifest=json.loads(z.read('reproducibility_manifest.json'))
original=t; d=json.loads((P/'diagnostics.json').read_text()); b=json.loads((P/'basis_prior_corrected/basis_results.json').read_text()); res=json.loads((F/'final_results.json').read_text()); selection=json.loads((P/'selected_examples.json').read_text())
def replace(a,b):
    global t
    assert a in t,a[:100]
    t=t.replace(a,b)
def para(prefix,new):
    global t
    st=t.index(prefix); en=t.find('\n\n',st); assert en>st
    t=t[:st]+new+t[en:]
for a,c in [('5814','5810'),('1935','1882'),('4065','4118'),('1885','1824'),('2038200000','2042200000'),('2038202231','2042203322'),('2038201952','2042200886'),(r'5.32\times10^{-3}',r'5.10\times10^{-3}'),(r'8.52\times10^{-3}',r'8.21\times10^{-3}'),('0.00532','0.00510'),('0.00241','0.00212'),('0.01970','0.01767'),('0.00934','0.00877'),('0.00852','0.00821'),('0.00254','0.00223'),('0.02782','0.02454'),('0.01367','0.01286'),('0.04131','0.04426'),('0.00912','0.00905'),('0.00285','0.00253'),('0.00823','0.00783'),('1.69--923.52','1.91--1105.41'),('54.77','54.43'),('0.13552','0.13510'),('0.10337','0.10008'),('0.01190','0.01144'),('0.01786','0.01779')]:
    t=t.replace(a,c)
replace('A coefficient head adjusts the coefficients of a fixed, training-derived signal basis using observation-derived noise and phase features.','A coefficient head adjusts a training-derived signal basis using observation-derived noise and phase features. A multiscale convolutional module then reconstructs residual structure in physical coordinates, with supervision that follows the decaying envelope.')
replace(r'\widehat u&=Bc+(I-P_0)b,\qquad'+'\n'+r'\widehat h=W^{-1}\widehat u.\nonumber',r'u_0&=Bc+(I-P_0)b,\qquad h_0=W^{-1}u_0.\nonumber')
replace(r'this yields $\widehat u=b$ to floating-point precision.',r'this yields $u_0=b$ to floating-point precision.')
replace('The complete stored model has 9,589,685 parameters.','The waveform estimator through this head has 9,589,685 parameters.')
para(r'\caption{DGF architecture.',r'''\caption{DGF architecture. (a) Time--frequency and signed temporal representations feed a shared encoder and waveform decoder. (b) Mask-aware and joint-channel refiners, the signal-basis adapter and conditioned coefficient head produce an intermediate estimate. (c) A conditioned one-dimensional U-Net predicts a residual in physical coordinates. Skip connections join matching temporal resolutions; the numbers give feature-channel widths. A fixed frequency taper restricts the learned residual before addition to the intermediate estimate. The miniature waveforms and feature maps are schematic.}\label{fig:pipeline}\label{fig:architecture}
\end{figure*}''')
addition=r'''\subsection{Multiscale residual reconstruction}\label{sec:multiscale}
The coefficient representation captures the joint waveform structure. A one-dimensional U-Net~\cite{Ronneberger2015UNet} supplements it with a learned residual evaluated at the original sampling rate. Its inputs combine the observation and intermediate prediction in whitened and physical coordinates, the predicted analytic envelopes, the gap mask and normalized time. These quantities form 12 temporal channels. A 470-dimensional context vector uses the features in Table~\ref{tab:head_features}, recomputed with $y=u_0$.

The contracting path has feature widths 32, 64, 96 and 128 at lengths 1056, 528, 264 and 132. Each block contains two kernel-five convolutions with padding two, eight-group normalization and GELU. Factor-two average pooling connects successive levels. The expanding path uses linear upsampling, concatenation with the matching contracting-path features, and blocks of widths 96, 64 and 32. At the bottleneck, a $470\to192\to256$ context network supplies channel-wise scale and offset parameters. A kernel-one output layer predicts two residual series.

Writing $U_\theta$ for this output and $\sigma_h$ for the RMS of the intermediate physical waveform, the final estimate is
\begin{equation}
\widehat h=h_0+0.1\sigma_h\,\mathcal P_f U_\theta.\label{eq:multiscale_output}
\end{equation}
The Fourier multiplier of $\mathcal P_f$ is unity up to $0.3\,\mathrm{Hz}$, follows a half-cosine taper between $0.3$ and $0.4\,\mathrm{Hz}$, and is zero above $0.4\,\mathrm{Hz}$. This fixed operation acts on the learned residual. The output layer is initialized to zero, preserving $h_0$ at initialization. The residual module has 646,658 parameters, giving 10,236,343 parameters for the complete estimator. Appendix~\ref{app:residual_details} gives the normalization and conditioning conventions.

'''
replace(r'\subsection{Training objective and data partition}',addition+r'\subsection{Training objective and data partition}')
replace('Final two-channel optimization uses','The reconstruction objective is')
replace(r'\mathcal L=\mathcal M_{AE}+0.2E_S+0.02E_{\rm time,gap},',r'''\begin{split}
\mathcal L={}&\mathcal M_{AE}+0.2E_S+0.02E_{\rm time,gap}\\
&+a_e\frac{\rho^2}{\rho^2+100}(0.04E_{\rm env}+0.03E_\phi),
\end{split}''')
replace('The objective combines mismatch, normalized noise-weighted waveform error and normalized time-domain waveform error, with a factor-three weight inside the gap. Losses and population mismatches are arithmetic means over sources.',r'The first three terms combine direct mismatch with noise-weighted and physical-time residual errors, with a factor-three weight inside the gap. The envelope-normalized Huber term $E_{\rm env}$ and circular-phase term $E_\phi$ resolve the decaying waveform after its joint envelope peak. They use samples where the target envelope exceeds 1\% of its channel maximum. Their weight grows as $a_e=\min(1,e/10)$ at training epoch $e$ and is moderated by $\rho^2/(\rho^2+100)$. Appendix~\ref{app:residual_details} defines both terms. The objective is averaged over training observations; evaluation means give equal weight to each source.')
para(r'Figure~\ref{fig:training_workflow} summarizes training.',r'''Figure~\ref{fig:training_workflow} summarizes training. Each of the 60,000 physical training sources contributes four cached observation views. The first two have independently sampled variable gap widths around the same center and independent noise realizations; the third has a 33-s gap and another independent noise realization. The second and third views receive a common input--target sign drawn from $\{-1,1\}$ with equal probability. The fourth uses a variable gap and $n_{\rm mix}=n_0\cos\vartheta+n_1\sin\vartheta$, with $\vartheta$ uniform on $[0,2\pi)$ and $n_0,n_1$ independent noise realizations with the same instrument-noise covariance. This mixture preserves the Gaussian noise covariance. Widths, noise and signs are fixed across epochs, and all transform features are computed from the corresponding observations. Each residual-training epoch uses all 240,000 cached observations and 24,000 additional draws from training sources with $\rho<20$, for a total of 264,000 observations.''')
para(r'\caption{Simulation and training workflow.',r'''\caption{Simulation and training workflow. The source populations and noise calibration are separate. Supervised waveform estimation combines the dual branches, joint-channel synthesis and multiscale residual reconstruction. Four observation views per source vary the noise and gap conditions. Validation controls learning rates, early stopping and estimator selection. The complete estimator is frozen before independent testing. Dashed arrows show validation feedback; the miniature drawings are schematic.}\label{fig:training_workflow}
\end{figure*}''')
start=t.index(r'\subsection{Training protocol}'); end=t.index(r'\section{Reconstruction results}',start)
t=t[:start]+r'''\subsection{Training protocol}\label{sec:training}
Training separates waveform estimation from residual reconstruction. The dual-branch encoder, decoder and coefficient modules are adapted to the physical $A/E$ population as described in Appendix~\ref{app:initialization}. Their parameters remain fixed while the multiscale residual module learns from cached intermediate predictions and observation-derived features. AdamW uses batches of 128, initial learning rate $2\times10^{-4}$, weight decay $10^{-4}$ and gradient-norm clipping at one. A validation plateau of 15 epochs reduces the learning rate by a factor of 0.7, with a floor of $10^{-6}$.

A 12,000-source training subset initializes the residual module, followed by training on all 60,000 sources. The full stage has a maximum of 400 epochs, a minimum of 80 and early-stopping patience of 80, with an absolute improvement threshold of $10^{-6}$. Raw and exponential-moving-average weights, with decay 0.995, are evaluated on all 1,500 validation sources. Selection minimizes their mean direct mismatch subject to validation phase and envelope-error tolerances specified in Appendix~\ref{app:residual_details}. The complete estimator is fixed before generating the 6,000 independent test sources. All population metrics use the original time and phase, the common PSD and the same source population.

'''+t[end:]
replace(r'\caption{Empirical cumulative distribution functions of time-domain and noise-weighted mismatch for the complete test population. The logarithmic axes show the full range of reconstruction errors.}',r'\caption{Empirical cumulative distributions of time-domain and noise-weighted mismatch for all 6,000 test sources and the 5810-source $\rho\ge10$ population. The logarithmic axes show the full range of errors in each population.}')
para(r'We also characterize gap severity by',r'We also characterize gap severity by $\eta=\sum_{c,j}m_jh_{c,j}^2/\sum_{c,j}h_{c,j}^2$ and $T_{\rm gap}/\tau_{220}$. In the $\rho\ge10$ population, their medians are 0.282 and 0.242, and their 95th percentiles are 0.662 and 1.50. The 2695 sources with $\eta<0.25$ have mean mismatch 0.00472; the 123 with $\eta\ge0.75$ have mean 0.01837.')
para(r'Figure~\ref{fig:examples} shows two',r'''Figure~\ref{fig:examples} shows two $\rho\ge10$ sources reconstructed across 40- and 48-s gaps, with noise-weighted mismatches $2.60\times10^{-4}$ and $5.69\times10^{-4}$. The gaps contain 41.2\% and 45.5\% of the respective joint signal squared norms. Their gap-local relative errors are 4.6\% and 5.5\%, and both channels retain the decaying oscillations through the missing intervals. The windows extend for 3.2 and 5.6 dominant-mode damping times after the joint envelope peak.
\begin{figure*}[tp]\centering\includegraphics[width=0.885\textwidth]{latest_examples.pdf}
\caption{Waveform reconstruction for two $\rho\ge10$ sources across 40- and 48-s gaps in the $A$ (left) and $E$ (right) channels. Each panel superposes the noisy observation $d=Gh+n$ (gray), clean signal $h$, gapped clean signal $Gh$ and DGF reconstruction $\widehat h$ over the full 528-s record; insets resolve the gap. Shading marks signal suppression, with noise retained. All curves in both channels use the common scale $H_{\rm TDI}=\max_{c,j}|h_{c,j}|$ and connect the original 0.5-s samples. Source seeds are 2042203322 and 2042200886. Appendix~\ref{app:model_analysis} gives the example criteria.}\label{fig:examples}\par\vspace{8pt}
\includegraphics[width=0.885\textwidth]{latest_phase.pdf}
\caption{(a) Absolute phase-error distributions for all 6,000 sources in each channel. (b) Clean analytic phase and reconstructed phase on the clean-phase branch for the $A$ channel of source 2042203322, also shown in Fig.~\ref{fig:examples}(a). The displayed clean-signal envelope exceeds 10\% of its peak. The clean phase is $\phi_c=\operatorname{unwrap}(\arg z_c)$; the displayed reconstruction is $\phi_c+\arg(\widehat z_cz_c^*)$. Unwrapping removes adjacent jumps greater than $\pi$. Shading marks the gap and the dotted line marks the clean envelope peak.}\label{fig:phase}\end{figure*}''')
for a,c in [(r'$3.84^\circ$ in $A$ and $3.79^\circ$ in $E$',r'$3.73^\circ$ in $A$ and $3.80^\circ$ in $E$'),(r'$2.19^\circ$ and $2.14^\circ$',r'$2.19^\circ$ and $2.17^\circ$'),(r'$12.72^\circ$ and $11.58^\circ$',r'$11.56^\circ$ and $11.97^\circ$'),(r'$4.92^\circ$, $3.89^\circ$ and $11.61^\circ$',r'$4.90^\circ$, $3.81^\circ$ and $11.17^\circ$'),(r'$5.51^\circ$',r'$5.47^\circ$'),(r'$4.92^\circ$',r'$4.90^\circ$')]:replace(a,c)
replace('A shared encoder and joint-channel refinement estimate the clean LISA $A/E$ waveforms from the noisy observation and gap mask.','A shared encoder, joint-channel synthesis and multiscale residual reconstruction estimate the clean LISA $A/E$ waveforms from the noisy observation and gap mask.')
replace('DGF achieves lower mismatch than regularized fitting in the same basis by estimating the coefficients from the combined temporal, phase and noise features.','DGF combines learned coefficients with a multiscale residual to recover the waveform from temporal, phase and noise features. Its mean mismatch is lower than that of the regularized basis fit on the same observations.')
replace(r'The selected checkpoint has SHA-256 prefix \texttt{0ffce08b8856};',r'The waveform-estimator and residual-module checkpoints have SHA-256 prefixes \texttt{0ffce08b8856} and \texttt{e504aae2ba10}, respectively;')
replace(r'The computational source version is \texttt{ae47cce8a5c0}.',r'The computational source version is \texttt{96a8ea344c45}.')
para(r'The examples in Fig.~\ref{fig:examples} have',r'The examples in Fig.~\ref{fig:examples} are chosen for clear oscillations and small waveform residuals. Both have the joint envelope peak in the gap, $\rho\ge10$, more than three $\tau_{220}$ of post-peak coverage and at least six resolved extrema per channel. Their gap-local relative errors are below 6\% and their channel peak-phase errors below $3.5^\circ$. The source seeds are given in the caption; population statistics include every test source.')
replace(r'The final-head initialization has identifier \texttt{a1e9959a7a95}; final optimization uses Eq.~\eqref{eq:loss} and Sec.~\ref{sec:training}, followed by new independent testing.',r'The conditioned coefficient head starts from weights \texttt{a1e9959a7a95}. It is fitted with the waveform estimator fixed and then jointly with the decoder and refinement modules, using $\mathcal M_{AE}+0.2E_S+0.02E_{\rm time,gap}$. This gives the intermediate estimator $u_0$ used in Sec.~\ref{sec:multiscale}. Its configuration and checkpoint are recorded in the manifest.')
addition=r'''\subsection{Residual-module conventions}\label{app:residual_details}
Let $\sigma_0$ be the arithmetic mean of the two channel noise standard deviations obtained by integrating the calibrated PSD. Physical-coordinate arithmetic in the residual module uses $\bar h_0=h_0/\sigma_0$ and $\bar d=d/\sigma_0$. Set $r=\max[\operatorname{RMS}(u),10^{-4}]$, $s_h=\max[\operatorname{RMS}(\bar h_0),10^{-5}]$ and $s_d=\max[\operatorname{RMS}(\bar d),10^{-5}]$. The 12 input channels are
\begin{equation}
\left(\frac{u}{r},\frac{u_0}{r},\frac{\bar h_0}{s_h},\frac{\bar d}{s_d},\frac{|\mathcal A[\bar h_0]|}{s_h},m,\frac{t}{528\,\mathrm{s}}\right),
\end{equation}
where $\mathcal A[x]=x+i\mathcal H[x]$ and each of the first five entries contains two channels. Thus $\sigma_h=\sigma_0s_h$ in Eq.~\eqref{eq:multiscale_output}. The context vector is recomputed from $u,u_0,m$ and the physical-frequency CWT using the feature definitions in Table~\ref{tab:head_features}. The context network has one hidden GELU layer and produces $(\gamma,\beta)\in\mathbb R^{128}\times\mathbb R^{128}$. It transforms bottleneck features $D$ as $D[1+0.1\tanh\gamma]+0.1\beta$, broadcasting along time. Convolutions use zero padding; upsampling uses linear interpolation with \texttt{align\_corners=False}.

The envelope and phase terms use scaled targets $y=h/\sigma_0$, predictions $\widehat y=\widehat h/\sigma_0$, and analytic series $z=\mathcal A[y]$, $w=\mathcal A[\widehat y]$. Write $A_{c,j}=|z_{c,j}|$, $A_{c,\max}=\max(\max_j A_{c,j},10^{-6})$ and $j_\star=\arg\max_j\sum_c A_{c,j}^2$. Let $R_{c,j}$ indicate $j\ge j_\star$ and $A_{c,j}\ge0.01A_{c,\max}$. Then
\begin{align}
E_{\rm env}&=\frac{\sum_{c,j}R_{c,j}\,\ell_{0.5}\!\left(\frac{\widehat y_{c,j}-y_{c,j}}{A_{c,j}+0.03A_{c,\max}}\right)}{\max(\sum_{c,j}R_{c,j},1)},\nonumber\\
E_\phi&=\frac{\sum_{c,j}\omega_{c,j}[1-\operatorname{Re}(w_{c,j}z_{c,j}^*)/D_{c,j}]}{\max(\sum_{c,j}\omega_{c,j},10^{-6})},\label{eq:envelope_losses}
\end{align}
where $\ell_{0.5}(x)=x^2$ for $|x|<0.5$ and $|x|-0.25$ otherwise, $\omega_{c,j}=R_{c,j}\sqrt{A_{c,j}/A_{c,\max}}$, and
\begin{equation}
\begin{gathered}
d_{c,j}=\sqrt{|w_{c,j}|^2+10^{-6}A_{c,j}^2}\max(A_{c,j},10^{-6}),\\
D_{c,j}=\max(d_{c,j},10^{-12}).
\end{gathered}
\end{equation}
Clean targets supply the supervised envelopes and SNR. Inference obtains envelopes from $h_0$.

Validation selection combines the complete-population mismatch with phase and envelope-error constraints. Relative to the intermediate estimator, the channel peak-phase means and $\rho\ge10$ mismatch may increase by at most 0.5\%; envelope-weighted phase errors in the 1\%--10\% and 0.1\%--1\% post-peak amplitude bands may increase by at most 1\%, while the relative waveform error in the 1\%--10\% band is bounded by its intermediate-estimator value to numerical tolerance. All constraints use the 1,500 validation sources. Model and checkpoint selection precede independent test generation.

'''
replace(r'\subsection{Periodograms}',addition+r'\subsection{Periodograms}')
replace('0.00438 and 0.00715','0.00408 and 0.00694')
for name,family in [('33 s','fixed66'),('50 s','fixed100')]:
    for label,key in [(r'$\rho\ge10$','snr_ge10'),('all','psd')]:
        vals=res['results'][family]['optimized'][key]
        t,n=re.subn(re.escape(name+' / '+label)+r' & [^\n]+',lambda _:name+' / '+label+' & '+' & '.join(f'{vals[k]:.5f}' for k in ['mean','median','p95'])+r'\\',t); assert n==1
para('A reference DGF configuration with joint-channel synthesis',r'To assess multiscale residual reconstruction, we evaluate the intermediate estimator $h_0$ on the same independent observations. For variable gaps, its complete-population mean mismatch is 0.00903, compared with 0.00821 for the full DGF model. The paired difference $\mathcal M_{\rm DGF}-\mathcal M_{h_0}$ has mean $-8.21\times10^{-4}$ and a 95\% percentile bootstrap interval $[-9.46\times10^{-4},-6.96\times10^{-4}]$, using 5,000 source-level resamples. The corresponding $\rho\ge10$ means are 0.00578 and 0.00510. This comparison holds the waveform estimator, observations and evaluation weights fixed while adding the residual module.')
replace(r'\section{Fixed-duration gaps and reference comparison}',r'\section{Fixed-duration gaps and multiscale reconstruction}')
replace('a comparison with a reference DGF configuration','comparisons with the intermediate estimator and regularized basis fitting')
replace('We compare DGF with a linear estimator using the same rank-128 basis $B$.','We compare DGF with a linear estimator using the rank-128 basis $B$ of its coefficient module.')
replace(r'$-0.04341$, with 95\% percentile bootstrap interval $[-0.04532,-0.04156]$',r'$-0.04744$, with 95\% percentile bootstrap interval $[-0.04938,-0.04545]$')
replace('& Basis fit & 0.04426 & 0.02030 & 0.14912','& Basis fit & 0.04426 & 0.02172 & 0.16585')
replace('& Basis fit & 0.05193 & 0.02141 & 0.21892','& Basis fit & 0.05565 & 0.02279 & 0.22817')
replace(r'0.03690, 0.00410 and $1.65\times10^{-4}$',r'0.03926, 0.00426 and $1.55\times10^{-4}$')
replace(r'0.03322, 0.00319 and $1.11\times10^{-4}$',r'0.03491, 0.00316 and $9.42\times10^{-5}$')
bib+='\n'+r'''@article{Ronneberger2015UNet,
 author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
 title={{U-Net}: Convolutional Networks for Biomedical Image Segmentation},
 journal={arXiv e-prints}, eprint={1505.04597}, archivePrefix={arXiv}, primaryClass={cs.CV}, year={2015}
}
'''
t=t.replace('A regularized fit in the same training-derived signal basis','A regularized fit in the training-derived signal basis')
(O/'DGF.tex').write_text(t,encoding='utf8',newline='\n'); (O/'DGF_references.bib').write_text(bib,encoding='utf8',newline='\n')
(P/'manuscript.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),t.splitlines(True))),encoding='utf8',newline='\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest['schema_version']=2; manifest['computational_source_commit']='96a8ea344c457c31eb93a0412d4d94b0da1be5b3'
manifest['assets']['intermediate_estimator']=manifest['assets'].pop('final')
ck=F/'full/tail_unet/best.pth'; manifest['assets']['final_residual_module']=dict(path=str(ck.relative_to(R)).replace('\\','/'),sha256=sha(ck),metadata=dict(variant='tail_unet',epoch=5,score=.007174841674042762))
manifest['inference']=dict(components=['intermediate_estimator','final_residual_module'],formula='h_hat = W^-1 u0 + 0.1 sigma_h P_f U_theta(inputs, context)',residual_parameters=646658,total_parameters=10236343,frequency_multiplier=dict(unit_through_hz=.3,half_cosine_to_hz=.4,zero_above_hz=.4),normalization='shared physical scale sigma0; exact normalization in manuscript Appendix A and prd_tail/model.py')
manifest['test_seeds']=dict(first=2042200000,last=2042205999,n=6000)
manifest['figure_examples']=selection; manifest['figure_examples']['checkpoint_sha256']=sha(ck)
manifest['figure_examples']['source_data_sha256']={str(f.relative_to(F)).replace('\\','/'):sha(f) for f in [F/'fresh_test/truth.npy',F/'fresh_test/noise.npy',F/'fresh_test/seeds.npy',F/'variable_optimized_prediction.npy']}
manifest['linear_benchmark']=b; manifest['additional_diagnostics']={k:v for k,v in d.items() if k!='candidates'}
manifest['test_results']={k:v['optimized'] for k,v in res['results'].items()}; manifest['paired_intermediate_comparison']={k:v['paired_delta'] for k,v in res['results'].items()}
manifest['residual_training']=dict(source_n=60000,views=4,observations_per_epoch=264000,additional_training_draws=dict(n=24000,population='rho<20',replacement=True),objective='M_AE + 0.2 E_S + 0.02 E_time_gap + min(1,epoch/10)*rho^2/(rho^2+100)*(0.04 E_env+0.03 E_phase)',optimizer='AdamW',learning_rate=.0002,weight_decay=.0001,batch_size=128,lr_plateau=dict(factor=.7,patience=15,min_lr=1e-6),early_stopping=dict(min_epochs=80,max_epochs=400,patience=80,improvement=1e-6),ema=.995,selected_epoch=5,finished_epoch=85,selection_test_independent=True)
manifest['source_file_sha256']={str(f.relative_to(R)).replace('\\','/'):sha(f) for f in sorted((R/'prd_tail').glob('*.py'))}
manifest['figure_sha256']={f.name:sha(f) for f in O.glob('*.pdf') if f.name in ['architecture_dgf_20261015.pdf','training_workflow_dgf_20261015.pdf','latest_mismatch.pdf','latest_snr_gap.pdf','latest_examples.pdf','latest_phase.pdf','tdi_log_amplitudes_20261015.pdf','latest_residual_psd.pdf']}
(O/'reproducibility_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8',newline='\n')
print('Manuscript updated',len(t),'characters')
