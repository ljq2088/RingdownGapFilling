from pathlib import Path
from zipfile import ZipFile
import shutil,subprocess
P=Path(__file__).resolve().parent
root=Path('/home/ljq/code/RingdownGapFilling_PRD')
assert subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip()=='codex/mask-aware-ringdown-optimization'
dest=root/'docs/prd_snr_focus_20261015';dest.mkdir(exist_ok=True)
for p in P.iterdir():
    if p.suffix in ('.py','.json','.patch') or p.name in ('overleaf_final.pdf','overleaf_snr_focus.jpg'):
        shutil.copy2(p,dest/p.name)
shutil.copy2(P/'source/DGF.tex',dest/'DGF.tex')
with ZipFile(P/'overleaf_before.zip') as z:(dest/'DGF_before.tex').write_bytes(z.read('DGF.tex'))
(dest/'README.txt').write_text('''SNR-focused manuscript revision, 15 October 2026

Overleaf: https://cn.overleaf.com/project/6ac83816750c2cce452054d2
Cloud build: 1a13cc85d7d-4f50f4e8d0564c83

The abstract, Introduction evaluation paragraph, Results, Discussion, captions and Appendix B now lead with the rho >= 10 results while retaining the full 6000-source statistics and distributions. Table I and the fixed-duration table show the rho >= 10 rows first. Peak-phase, amplitude, gap-location and reference-comparison summaries explicitly identify their full-population scope. The low-SNR subpopulation is not discussed separately.

All numerical table rows are unchanged; only row order and captions changed. All equations, bibliography records and eight figure files are unchanged. Frozen checkpoint and data are unchanged. The cloud PDF has 12 pages, all figures precede Appendix A, and no external link occurs on figure pages. verification.json records checks and hashes.

Only DGF.tex changed in the cloud source. The full before/after source ZIPs are retained in local outputs/snr_focus_20261015. Active training code, data, checkpoints and processes were not modified.
''',encoding='utf-8')
print(dest)
