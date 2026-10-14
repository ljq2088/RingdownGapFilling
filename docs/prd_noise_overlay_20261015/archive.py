from pathlib import Path
import shutil,subprocess
HERE=Path(__file__).resolve().parent;ROOT=Path('/home/ljq/code/RingdownGapFilling_PRD')
assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='codex/mask-aware-ringdown-optimization'
dest=ROOT/'docs/prd_noise_overlay_20261015';dest.mkdir(exist_ok=True)
for p in HERE.glob('*'):
 if p.suffix in ['.py','.json','.npz','.patch'] or p.name=='overleaf_fig4.jpg':shutil.copy2(p,dest/p.name)
for name in ['DGF.tex','latest_examples.pdf','latest_phase.pdf','tdi_log_amplitudes_20261015.pdf']:
 shutil.copy2(HERE/'source'/name,dest/name)
(dest/'README.md').write_text('''# Restore native noisy-observation overlay and remove rank-example figure

Overleaf project: `6ac83816750c2cce452054d2`. The fresh cloud source archive and compiled PDF match the locally verified edit.

Figure 4 overlays the original observation d=Gh+n in gray with the clean, gapped-clean and reconstructed waveforms in the same axes. All four use the same per-source amplitude normalization. The noise multiplier is exactly one; every observed sample fits inside the displayed ordinate limits. The separate upper observation strips have been removed.

The two illustrated sources have seeds 2038202231 and 2038201952, with gaps of 36 and 45 seconds and direct mismatches 0.0005587358 and 0.0006072640. Their full-record and gap relative errors are below 6%, and both channels have envelope-peak phase errors below 2.5 degrees. Native curvature, visible oscillations and the original noise-to-signal peak ratio enter the illustration criteria. No smoothing, interpolation or phase/time alignment is applied. The phase example and logarithmic-amplitude figure use the same sources.

The previous Figure 8, containing the median, 90th-percentile and maximum-mismatch rank examples, has been removed from the manuscript with its discussion, cross-references and cloud asset. Diagnostic arrays and earlier sources remain in the local audit history. The former Figure 9 logarithmic-amplitude comparison is now Figure 8.

All result tables and the complete-population mismatch, SNR/gap and residual-PSD plots are byte-for-byte unchanged. Removing an optional illustrative diagnostic does not change the 6000-source population or its reported performance. Training continues independently using the frozen prd_tail sources.

The updated manuscript has 12 pages and 8 numbered figures. All figure pages have zero external hyperlinks. The bibliography boundary introduced by the preceding cleanup remains in place.
''',encoding='utf-8')
print(dest)
