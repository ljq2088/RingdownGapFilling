import os, json, subprocess, zipfile, time
from prd_physical.common import write, sha
from .settings import *

def main():
    for gate in ['preflight.json','final_smoke_gate.json','benchmark_a.json','benchmark_b.json']:
        assert json.loads((OUT/gate).read_text())['passed'],gate
    files=[]
    for directory in ['prd_lowrank','prd_strategy','prd_frequency','prd_optimize','prd_physical','model','utils','config']:
        files.extend((ROOT/directory).rglob('*.py'))
    manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}
    path=OUT/'runtime_source_manifest.json'
    if path.exists():assert json.loads(path.read_text())==manifest,'frozen runtime changed'
    else:
        write(path,manifest)
        with zipfile.ZipFile(OUT/'runtime_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(set(files)):z.write(p,str(p.relative_to(ROOT)))
    assets=[PROBE/'frequency_snapshot.pth',PROBE/'training_direction_basis.npz',PROBE/'frequency_snapshot_validation_prediction.npy',CAL]
    write(OUT/'frozen_asset_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in assets})
    write(OUT/'training_config.json',dict(variants=list(VARIANTS),train_unique=TRAIN_N,validation_unique=VAL_N,test_unique=TEST_N,observations_per_source=VIEWS,training_observations_per_epoch=TRAIN_N*VIEWS,rank=RANK,additional_trainable_parameters=66688,backbone_checkpoint_sha256=SNAPSHOT_SHA,basis_sha256=sha(PROBE/'training_direction_basis.npz'),calibration_sha256=sha(CAL),batch_sources=BATCH,max_epochs=MAX_EPOCHS,min_epochs=MIN_EPOCHS,patience=PATIENCE,min_delta=DELTA,learning_rate=2e-4,min_learning_rate=2e-6,plateau_patience=15,ema_decay=EMA,training_seed=SEED,new_noise_seed_range=[NOISE_SEED,NOISE_SEED+(VIEWS-1)*TRAIN_N-1],fresh_test_seed_range=[TEST_SEED,TEST_SEED+TEST_N-1],primary_metric='direct signed unmaximized joint A/E PSD mismatch arithmetic mean',objective='direct overlap + .5 spectral relative MSE + .05 physical time/gap relative MSE + .02 Hann regularizer',observation='G*h+n',random_recrop=False,all_rows_retained=True,cpu_threads_per_worker=2,cuda_used=False,minimum_available_ram_gib=MIN_RAM_GIB,resume='atomic epoch checkpoint: raw/EMA/best weights, optimizer, scheduler, torch/shuffle RNG',preprocessing='frozen physical PSD whitening; both CWT banks and db4 bands recomputed for fresh noise'))
    old=OUT/'launcher.json'
    if old.exists():
        pid=json.loads(old.read_text())['pid'];p=__import__('pathlib').Path('/proc')/str(pid)/'cmdline'
        if p.exists() and b'prd_lowrank.run' in p.read_bytes():print(old.read_text());return
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMBA_NUM_THREADS='1')
    with open(OUT/'pipeline.log','ab',buffering=0) as log:
        p=subprocess.Popen(['/home/ljq/miniconda3/envs/gw_ml_env/bin/python','-u','-m','prd_lowrank.run'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    v=dict(pid=p.pid,started_unix=time.time(),branch='codex/mask-aware-ringdown-optimization',device='cpu',execution='detached coordinator; parallel full-population heads; automatic selection and new independent test')
    write(old,v);print(json.dumps(v),flush=True)
if __name__=='__main__':main()
