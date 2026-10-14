from pathlib import Path
import shutil,subprocess
here=Path(__file__).resolve().parent;root=Path('/home/ljq/code/RingdownGapFilling_PRD')
assert subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip()=='codex/mask-aware-ringdown-optimization'
dest=root/'docs/prd_main_text_figures_20261015';dest.mkdir(exist_ok=True)
for p in here.glob('*'):
 if p.suffix in ['.py','.json','.patch'] or p.name=='overleaf_main_figures.jpg':shutil.copy2(p,dest/p.name)
shutil.copy2(here/'source/DGF.tex',dest/'DGF.tex')
(dest/'README.md').write_text('''# Integrate supporting figures into the main Results section

Overleaf project: `6ac83816750c2cce452054d2`.

The SNR/gap-location diagnostic, previously Figure 7 in Appendix B, is now Figure 4 in the main Results section. The logarithmic waveform-amplitude comparison, previously Figure 8 in Appendix B, is now Figure 7 in the main amplitude/spectral results. The signed full-record comparison is now Figure 5, and the phase comparison remains Figure 6. Residual PSDs are now Figure 8.

Results are organized as mismatch, gap location/waveform recovery, phase recovery, and amplitude/spectral fidelity. Signed waveform/phase plots and logarithmic-amplitude/PSD plots share paired float pages with separate numbered captions. The signed/phase display width is 0.885 of the text width; every original PDF figure file is unchanged. The global numerical results, sample selection, result tables, trained checkpoint and test population are unchanged.

The supplementary-waveform appendix has been removed. Detailed sample/metric conventions join Appendix A (Implementation and reproducibility); Appendix B contains fixed-duration gap tests and the reference comparison. Cross-references resolve to these new locations.

The compiled manuscript has 12 pages. All 8 figures appear before the appendices, there are no blank pages, and no figure page contains an external hyperlink. Fresh cloud source/PDF verification is recorded in verification.json. Active model training files and processes were not modified.
''',encoding='utf-8')
print(dest)
