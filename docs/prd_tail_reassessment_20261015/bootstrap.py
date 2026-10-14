from pathlib import Path
import sys,json,hashlib,subprocess,os,time
ROOT=Path('/home/ljq/code/RingdownGapFilling_PRD');sys.path.insert(0,str(ROOT))
from prd_tail.settings import *
from prd_physical.common import sha,write
OUT.mkdir(exist_ok=True)
manifest={}
for name in ('runtime_source_manifest.json','asset_manifest.json'):
    manifest.update(json.loads((COHERENT/name).read_text()))
for path in (ROOT/'prd_tail').glob('*.py'):manifest[str(path.relative_to(ROOT))]=sha(path)
manifest[str(START.relative_to(ROOT))]=sha(START);manifest[str(CAL.relative_to(ROOT))]=sha(CAL)
write(OUT/'frozen_manifest.json',manifest)
protocol=dict(seed=SEED,variants=list(VARIANTS),training_sources=TRAIN_N,views=4,observations_per_epoch=264000,
 validation_sources=VAL_N,pilot_sources=PILOT_N,pilot_observations_per_epoch=52800,pilot_max=PILOT_MAX,pilot_min=PILOT_MIN,pilot_patience=PILOT_PATIENCE,
 full_candidates=2,full_max=FULL_MAX,full_min=FULL_MIN,full_patience=FULL_PATIENCE,delta=DELTA,ema=EMA,
 batch={'tail_basis':512,'tail_unet':128,'qnm_tail':512},primary='mean direct joint signed PSD-weighted mismatch, fixed time and phase',
 guardrails=['per-channel peak phase <= baseline*1.005','SNR>=10 mismatch <= baseline*1.005','1%-10% envelope local error <= baseline','1%-10% and 0.1%-1% envelope phase <= baseline*1.01'],
 data_policy='All 60000 physical sources retained. Four complete observation views and 10% added low-SNR training observations. No test data in model selection.',
 auxiliary_supervision='Training-only detector-frame remnant mass/spin for QNM branch. Inference uses only cached observation features and the baseline prediction.',
 final_test=dict(seed_start=TEST_SEED,count=TEST_N,untouched_until_selection_freeze=True),observation='G*h+n',baseline_sha256=sha(START),
 resources=dict(torch_threads=2,gpu_memory_fraction=.70,gpu_cache_budget_gib=5.2,external_gpu_reserve_gib=4.2,ram_guard_gib=5),
 response_scope='Existing isolated-QNM onset and TDI protocol retained; this run tests reconstruction error, not pre-merger robustness.',
 no_source_strain_reconstruction_claim=True)
write(OUT/'protocol.json',protocol)
# Track the controlled stop separately from unexpected training failures.
if (COHERENT/'failure.json').exists():
    failure=json.loads((COHERENT/'failure.json').read_text())
    if 'STOP requested' in failure.get('traceback',''):
        write(OUT/'previous_run_checkpoint_handoff.json',dict(reason='User requested reassessment of full-waveform and late-time fidelity',previous_failure=failure,
            previous_status=json.loads((COHERENT/'status.json').read_text()),last_checkpoint=str(COHERENT/'full/noise_mix/last.pth'),time=time.time()))
with open(OUT/'pipeline.log','ab',buffering=0) as log:
    p=subprocess.Popen([sys.executable,'-u','-m','prd_tail.run'],cwd=ROOT,stdout=log,stderr=log,start_new_session=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
write(OUT/'launched.json',dict(pid=p.pid,time=time.time(),python=sys.executable,root=str(ROOT),protocol=protocol))
print(json.dumps({'pid':p.pid,'output':str(OUT),'frozen_files':len(manifest)},indent=2))
