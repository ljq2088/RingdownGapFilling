from pathlib import Path
import zipfile,json,difflib,shutil,re
P=Path(__file__).resolve().parent
with zipfile.ZipFile(P/'overleaf_before.zip') as z: original=z.read('DGF.tex').decode('utf8')
s=original
changes=[]
def replace(old,new,reason):
    global s
    assert s.count(old)==1,(reason,s.count(old),old[:80])
    s=s.replace(old,new);changes.append(reason)
replace('We evaluate DGF on 6,000 independent simulated LISA observations with gaps spanning approximately 6\\%--10\\% of each record.',
        'We evaluate DGF on 6,000 independent simulated LISA $A/E$ observations, suppressing the signal over approximately 6\\%--10\\% of each record while retaining stationary Gaussian instrument noise throughout.',
        'State the tested observation conditions directly in the abstract')
replace('These results demonstrate recovery of the oscillatory waveform across gaps, including intervals containing the signal peak, within the simulated source population and stationary Gaussian instrument-noise model.',
        'These results demonstrate recovery of damped oscillations through intervals containing the signal peak in the simulated source population.',
        'Condense the abstract ending and remove a duplicated noise-model qualification')
replace('The time--frequency branch represents local oscillatory content through complex Morlet continuous wavelet transform (CWT) coefficients.',
        'The time--frequency branch uses complex Morlet continuous wavelet transform (CWT) features, including a second bank with explicit low-frequency coverage.',
        'Match the introductory description to the two actual CWT banks')
replace('A separate set of 128 noise realizations calibrates the coefficient head.',
        'The first 128 of the 512 noise-calibration realizations also calibrate the coefficient head.',
        'Correct the shared PSD/RMS calibration population, confirmed by bitwise reproduction')
replace('Both noise-calibration sets are independent of the source population.',
        'The calibration population is independent of the source data.',
        'Distinguish independence from source data and reuse between calibration estimators')
replace('These widths correspond to 32--52 s and preserve the reference interval center.',
        'These widths correspond to 32--52 s and preserve the reference interval center. Generation requires $0\\le a_L$ and $a_L+L\\le N$ and aborts if either condition fails. For all 60,000 training, 1,500 validation and 6,000 test sources, neither clipping operation is active and $132\\le a\\le263$; hence every allowed width fits within the window.',
        'Close the boundary definition and report the complete-population check')
replace('The Morlet parameter is $\\mu=13.4$. We evaluate scales $1,2,\\ldots,16$ using reflection padding and $L^1$ wavelet normalization, and retain every fourth time sample, giving a $2\\times16\\times264$ array per channel.',
        'The Morlet parameter is $\\mu=13.4$. The first bank uses scales $a=1,2,\\ldots,16$, with reflection padding and $L^1$ normalization. Its nominal center frequencies are $\\mu f_s/(2\\pi a)$: scales 5--16 peak between 0.853 and 0.267 Hz, while scales 1--4 have centers above Nyquist. Retaining every fourth time sample gives a $2\\times16\\times264$ feature array per channel.',
        'Explain the high-frequency legacy bank without attributing low-frequency coverage to it')
replace('A second Morlet CWT uses scales $\\mu f_s/(2\\pi f_r)$,',
        'The low-frequency adapter uses a second Morlet CWT at scales $\\mu f_s/(2\\pi f_r)$,',
        'Identify the explicit low-frequency input path')
replace(' These values quantify recovery at the original arrival time and phase.','',
        'Remove a repeated evaluation convention')
replace(' These conditional summaries include variation in source parameters, noise and gap duration.','',
        'Remove a repeated subgroup qualification')
replace(' Their contribution to a full-record error depends on its weighting.','',
        'Remove a generic error-weighting statement')
replace('Figure~\\ref{fig:phase}(b) compares analytic phases for the $A$ channel of one example.',
        'Figure~\\ref{fig:phase}(b) compares the clean analytic phase with the reconstruction expressed on the clean-phase branch for the $A$ channel of one example.',
        'Clarify that the phase plot uses a clean-referenced branch')
replace('(b) Analytic phase of the $A$ channel for source 2038202231, also shown in Fig.~\\ref{fig:examples}(a).',
        '(b) Clean analytic phase and reconstructed phase on the clean-phase branch for the $A$ channel of source 2038202231, also shown in Fig.~\\ref{fig:examples}(a).',
        'Specify the displayed phase quantity without implying an independent cycle-count test')
replace('Tests with continuous merger histories would assess sensitivity to the pre-reference ring-up entering the TDI delays. Tests with moving unequal-arm constellations and nonstationary noise would assess sensitivity to response and noise-model errors. Independently perturbed mode amplitudes, phases and frequencies would test how much structure outside the training relations is preserved.',
        'The next validation targets are the pre-reference signal history entering the TDI delays, errors in the response and noise models, and departures from the assumed mode-amplitude and phase relations. Application to missing TDI samples requires simulations with the corresponding observation and noise covariance.',
        'Condense repetitive future-work phrasing and state the application boundary')
anchor='Mixing along the temporal, feature and scale axes, followed by convolutional embedding and linear projection, produces $8\\times32\\times64$ features.\n'
replace(anchor,anchor+'\nFor the reflected 2048-sample FFT, the Morlet response at a sampled frequency $f_k$ is proportional to $\\widehat\\psi(2\\pi a f_k/f_s)$, with the Nyquist bin halved by the implementation. The maximum discrete gains at scales 1--3 are at most $3.6\\times10^{-4}$ of the continuous peak gain; scale 4 retains about 0.70 of that gain near Nyquist. This bank supplies predominantly high-frequency features. The accompanying manifest records each filter\'s peak frequency and gain.\n',
        'Record discrete-response verification, including the substantial scale-4 tail')
replace('The supplied legacy DGF weights initialize reproduction. Their training code uses pointwise squared and absolute errors on a separate 10,000-waveform simulation bank.',
        'We initialize the encoder and waveform decoder from the supplied DGF checkpoint with SHA-256 prefix \\texttt{e7dcf7e55cc6}. This checkpoint is a fixed input to the subsequent training protocol. Its associated bank contains 10,000 auxiliary single-series simulations with source total mass uniform over $[10^5,10^6]\\,M_\\odot$, mass ratio over $[0.5,0.8]$ and redshift over $[4,6]$. The archived training code assigns equal coefficients to pointwise squared and absolute errors, with unit mask weight. The physical training, validation and test populations are generated anew using the disjoint seed ranges given above.',
        'Identify the fixed initialization and document the auxiliary data and actual archived loss')
replace('Figure~\\ref{fig:examples} uses sources with the joint peak in the gap, $\\rho\\ge10$, at least four $\\tau_{220}$ of post-peak coverage and six visible extrema per channel. Full-record and gap-local errors are below 6\\%, and peak-phase errors below $2.5^\\circ$. The plotting manifest gives the channel-balance, native-sample curvature and noise-amplitude criteria.',
        'The sources in Fig.~\\ref{fig:examples} were chosen by inspecting native-sample waveform plots with the joint peak in the gap, $\\rho\\ge10$, at least four $\\tau_{220}$ of post-peak coverage and six visible extrema per channel. Full-record and gap-local errors below 6\\% and peak-phase errors below $2.5^\\circ$ also enter the example selection. Their seeds and normalization are given in the caption; the accompanying manifest records the additional display criteria.',
        'Distinguish example selection thresholds from population statistics without success-case labels')
# All numerical tables and core numbered metric/adapter/loss equations remain exact.
tables=lambda t:re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}',t,re.S)
assert tables(s)==tables(original)
for label in ('eq:basis_adapter','eq:refinement','eq:loss','eq:observation'):
    blocks=lambda t: [m.group() for m in re.finditer(r'\\begin\{(equation|align)\}.*?\\end\{\1\}',t,re.S) if label in m.group()]
    assert blocks(s)==blocks(original),label
(P/'source/DGF.tex').write_text(s,encoding='utf8',newline='\n')
(P/'manuscript.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='before/DGF.tex',tofile='after/DGF.tex')),encoding='utf8')
(P/'editorial_changes.json').write_text(json.dumps(changes,indent=2,ensure_ascii=False),encoding='utf8')
manifest=json.loads((P/'source/reproducibility_manifest.json').read_text())
audit=json.loads((P/'technical_audit.json').read_text())
manifest['noise_rms']['relationship_to_psd_calibration']='First 128 of the same 512 noise realizations, deterministically regenerated from identical seeds and crop; independent of source-data populations.'
manifest['noise_rms']['shared_between_calibration_estimators']=True
manifest['second_review_audit']=audit
manifest['initialization_provenance_status']='Fixed supplied checkpoint and associated dataset are identified by full hashes. Historical original random-split indices, optimizer state and exact original checkpoint-selection history are unavailable. Subsequent physical adaptation and final training have recorded validation selection.'
(P/'source/reproducibility_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf8')
shutil.copy2(P.parent/'reviewer_revision_20261015/figure_helpers.py',P/'figure_helpers.py')
draw=(P.parent/'reviewer_revision_20261015/redraw_methods.py').read_text(encoding='utf8')
draw='from figure_helpers import *\n\n'+draw[draw.index('fig,ax=canvas(8.20)'):]
draw=draw.replace('Separate 128: noise RMS','First 128: noise RMS')
(P/'redraw_training.py').write_text(draw,encoding='utf8')
stage=(P.parent/'reviewer_revision_20261015/stage_compile.py').read_text(encoding='utf8').replace('ringdown_reviewer_revision_20261015','ringdown_reviewer2_revision_20261015')
(P/'stage_compile.py').write_text(stage,encoding='utf8')
print('Applied',len(changes),'targeted revisions; all four numerical tables and core equations preserved.')
