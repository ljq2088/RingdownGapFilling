from pathlib import Path
import sys,json,subprocess,os
ROOT=Path('/home/ljq/code/RingdownGapFilling_PRD');sys.path.insert(0,str(ROOT))
from prd_coherent.settings import *
from prd_physical.common import write,sha
from prd_conditioned.common import check
check();OUT.mkdir(exist_ok=True)
protocol=dict(date='2026-10-15',primary='full-population direct signed joint A/E noise-weighted mismatch',selection='validation only; no time or phase alignment; preserve per-channel mean peak-phase and SNR>=10 mean mismatch within 0.5% of baseline',batch=256,baseline=str(START),baseline_sha=sha(START),variants=list(VARIANTS),pilot=dict(sources=PILOT_N,maximum_epochs=PILOT_MAX,minimum_epochs=PILOT_MIN,patience=PILOT_PATIENCE),full=dict(sources=TRAIN_N,maximum_epochs=FULL_MAX,minimum_epochs=FULL_MIN,patience=FULL_PATIENCE),ema=EMA,minimum_improvement=DELTA,all_sources_retained=True,training_views=3,noise_mix_extra_observations_fraction=.1,noise_mix_refresh_epochs=20,development_test_seeds=[2038200000,2038205999],untouched_final_test_seeds=[TEST_SEED,TEST_SEED+TEST_N-1],test_n=TEST_N,observation='G*h+n',gpu_fraction=.78,minimum_available_ram_gib=6,minimum_free_gpu_gib=4,checkpoints='last completed epoch includes optimizer, scheduler, raw+EMA models and RNG; rerun resumes',model_augmentation='independent equal-covariance Gaussian noise mixtures; recompute all features; no amplitude or time stretching',physics_unchanged=True)
write(OUT/'protocol.json',protocol)
sources=json.loads((OLD/'runtime_source_manifest.json').read_text());sources.update({str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'prd_coherent').glob('*.py')})
write(OUT/'runtime_source_manifest.json',sources)
assets=json.loads((OLD/'asset_manifest.json').read_text())
for path in (START,CAL,PROBE/'training_direction_basis.npz',OUT/'protocol.json',OUT/'mask_gram.npy'):
    assets[str(path.relative_to(ROOT))]=sha(path)
write(OUT/'asset_manifest.json',assets)
print(json.dumps(dict(output=str(OUT),sources=len(sources),assets=len(assets),protocol=protocol),indent=2))
