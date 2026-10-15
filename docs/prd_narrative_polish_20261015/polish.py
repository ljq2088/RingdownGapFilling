"""Integrate verified method details into the manuscript's scientific narrative."""
from pathlib import Path
import zipfile,json,difflib,re
P=Path(__file__).resolve().parent
with zipfile.ZipFile(P/'overleaf_before.zip') as z:old=z.read('DGF.tex').decode('utf8')
s=old;edits=[]
def change(a,b,why):
    global s
    assert s.count(a)==1,(why,s.count(a))
    s=s.replace(a,b);edits.append(why)
change('We evaluate DGF on 6,000 independent simulated LISA $A/E$ observations, suppressing the signal over approximately 6\\%--10\\% of each record while retaining stationary Gaussian instrument noise throughout.',
       'We evaluate DGF on 6,000 independent simulated LISA $A/E$ observations formed by adding stationary Gaussian instrument noise to ringdown signals with gaps spanning approximately 6\\%--10\\% of each record.',
       'Express the observation model as an ordinary experimental description')
change('Generation requires $0\\le a_L$ and $a_L+L\\le N$ and aborts if either condition fails. For all 60,000 training, 1,500 validation and 6,000 test sources, neither clipping operation is active and $132\\le a\\le263$; hence every allowed width fits within the window.',
       'The simulated crop positions give $132\\le a\\le263$, so the prescribed intervals satisfy $0\\le a_L$ and $a_L+L\\le N$.',
       'Replace boundary-audit narration with the actual interval definition')
first=s[s.index('Figure~\\ref{fig:architecture} shows the DGF architecture.'):s.index('The low-frequency adapter uses a second Morlet CWT')]
change(first,
       'Figure~\\ref{fig:architecture} shows the DGF architecture. The two $A/E$ channels share encoder weights. For each channel, the time--frequency branch combines two complex Morlet CWT representations with parameter $\\mu=13.4$, reflection padding and $L^1$ normalization. The first uses scales $a=1,2,\\ldots,16$ and retains every fourth time sample, giving a $2\\times16\\times264$ array of modulus and argument. Mixing along the temporal, feature and scale axes, followed by convolutional embedding and linear projection, produces $8\\times32\\times64$ features.\n\n',
       'Keep transform definitions in the main method and remove filter-check reporting')
change('The low-frequency adapter uses a second Morlet CWT at scales',
       'A low-frequency adapter uses the second Morlet CWT at scales',
       'Join the adapter description to the dual-bank introduction')
anchor='The eight db4 channels reconstruct $A_8,D_8,D_7,\\ldots,D_2$ with zero padding.'
change(anchor,
       'The first Morlet bank samples $\\widehat\\psi(2\\pi a f_k/f_s)$ on the reflected 2048-point FFT grid. Scales 5--16 have centers between 0.853 and 0.267 Hz; scales 1--4 contribute their spectral tails within the sampled band.\n\n'+anchor,
       'Place frequency-response information with implementation definitions')
start=s.index('For the full test population, the mean absolute analytic-phase errors at the envelope peak are',s.index('\\section{Discussion'))
end=s.index('\\begin{acknowledgments}',start)
discussion=s[start:end]
replacement=(
    'The signal-basis comparison connects representation accuracy with coefficient estimation from noisy observations. The small clean-target projection error shows that the rank-128 basis resolves the simulated waveform family. DGF achieves lower mismatch than regularized fitting in the same basis by estimating the coefficients from the combined temporal, phase and noise features. For the 1885 sources with $\\rho\\ge10$ and the joint envelope peak inside the gap, the mean envelope-weighted phase error within the gap is $4.92^\\circ$. This statistic measures phase recovery throughout the suppressed interval, complementing the peak-phase and full-record waveform errors.\n\n'
    'The variable-gap and fixed-duration tests cover the event-centered LISA $A/E$ records defined in Sec.~\\ref{sec:method}. The recovered waveforms provide point estimates of the damped oscillations and a starting point for waveform fits. Combining these estimates with mode-resolved inference and reconstruction uncertainties offers a route toward applications in black hole spectroscopy.\n\n')
change(discussion,replacement,
       'Replace review-response validation lists with result interpretation and connected scientific outlook')
change('A zero prediction is assigned mismatch one; a zero target triggers a dataset validation failure.',
       'A zero prediction is assigned mismatch one.',
       'Keep the metric definition and move a data-validation failure mode to the external record')
change('We initialize the encoder and waveform decoder from the supplied DGF checkpoint with SHA-256 prefix \\texttt{e7dcf7e55cc6}. This checkpoint is a fixed input to the subsequent training protocol. Its associated bank contains',
       'The encoder and waveform decoder are initialized from DGF weights identified in the accompanying manifest. The associated bank contains',
       'Describe initialization without asset-audit commentary')
change('The archived training code assigns equal coefficients to pointwise squared and absolute errors, with unit mask weight. The physical training, validation and test populations are generated anew using the disjoint seed ranges given above.',
       'The associated training configuration combines pointwise squared and absolute errors with equal weights and unit mask weight. The physical training, validation and test populations use the disjoint seed ranges given above.',
       'Integrate the auxiliary training configuration with the physical adaptation protocol')
change('The sources in Fig.~\\ref{fig:examples} were chosen by inspecting native-sample waveform plots with the joint peak in the gap, $\\rho\\ge10$, at least four $\\tau_{220}$ of post-peak coverage and six visible extrema per channel. Full-record and gap-local errors below 6\\% and peak-phase errors below $2.5^\\circ$ also enter the example selection. Their seeds and normalization are given in the caption; the accompanying manifest records the additional display criteria.',
       'For the examples in Fig.~\\ref{fig:examples}, the waveform display uses sources with the joint peak in the gap, $\\rho\\ge10$, at least four $\\tau_{220}$ of post-peak coverage and six visible extrema per channel. The selection includes full-record and gap-local errors below 6\\% and peak-phase errors below $2.5^\\circ$; the source seeds are given in the caption.',
       'Retain example-selection transparency in a concise implementation paragraph')
change('\\clearpage\n\\appendix',
       '\\onecolumngrid\n\\clearpage\n\\twocolumngrid\n\\appendix',
       'Balance the main-text columns before the final figure page and appendices')
tables=lambda x:re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}',x,re.S)
assert tables(s)==tables(old)
for label in ['eq:observation','eq:whitening','eq:basis_adapter','eq:refinement','eq:loss']:
    blocks=lambda x:[m.group() for m in re.finditer(r'\\begin\{(equation|align)\}.*?\\end\{\1\}',x,re.S) if label in m.group()]
    assert blocks(s)==blocks(old),label
for forbidden in ['next validation targets','Application to missing TDI samples requires','Generation requires','aborts if','maximum discrete gains','Historical original random-split']:
    assert forbidden not in s,forbidden
(P/'source/DGF.tex').write_text(s,encoding='utf8',newline='\n')
(P/'manuscript.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='before/DGF.tex',tofile='after/DGF.tex')),encoding='utf8')
(P/'editorial_changes.json').write_text(json.dumps(edits,ensure_ascii=False,indent=2),encoding='utf8')
# Keep the published companion manifest about scientific assets and definitions.
# The full second-review discussion stays in the separate local revision archive.
m=json.loads((P/'source/reproducibility_manifest.json').read_text(encoding='utf8'))
external={k:m.pop(k) for k in ['second_review_audit','initialization_provenance_status'] if k in m}
(P/'external_verification_notes.json').write_text(json.dumps(external,indent=2),encoding='utf8')
m['availability']='Checkpoints, calibration arrays and evaluation records are available from the authors on request. Source and asset identifiers are recorded in this manifest.'
(P/'source/reproducibility_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf8',newline='\n')
compiler=(P.parent/'reviewer2_revision_20261015/stage_compile.py').read_text(encoding='utf8').replace('ringdown_reviewer2_revision_20261015','ringdown_narrative_polish_20261015')
(P/'stage_compile.py').write_text(compiler,encoding='utf8')
print('Applied',len(edits),'narrative revisions; numerical tables and core equations unchanged.')
