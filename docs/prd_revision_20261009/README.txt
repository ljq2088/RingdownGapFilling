PRD revision workspace, 2026-10-09

Cloud project: https://cn.overleaf.com/project/6ac83816750c2cce452054d2
The unchanged original is https://cn.overleaf.com/project/69bb9d12f9a917663fb9b9c0.
See REVISION_REPORT.txt for scientific scope and remaining limitations.

Recompute from frozen predictions (does not train):
python FillingGap/evaluate_prd_revision.py --input-root /home/ljq/code/RingdownGapFilling --output FillingGap/results_prd_revision_20261009
python docs/prd_revision_20261009/verify_metric_conventions.py

Input arrays/checkpoints remain in the original checkout and are hashed in evaluation.json.
No time/phase alignment and no sample trimming are used.
To regenerate the edited manuscript, run build_revision.py in this directory.
Its source_before is the downloaded original, not local historical TeX/PDF.
The saved new source includes six vector figures; final cloud PDF/source verification is recorded in verification.json.
