"""Freeze reproducible sources and start the detached, resumable coordinator."""
import os,json,time,subprocess,zipfile
from pathlib import Path
from prd_physical.common import write,sha
from .settings import *
def main():
 for name in ['prepared.json','preflight.json','cache_gates.json','resource_gates.json','pilot.json']:
  assert json.loads((OUT/name).read_text())['passed'],name
 files=[]
 for directory in ['prd_frequency','prd_optimize','prd_physical','model','utils','config']:
  files.extend((ROOT/directory).rglob('*.py'))
 manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}
 f=OUT/'runtime_source_manifest.json'
 if f.exists():assert json.loads(f.read_text())==manifest,'frozen runtime changed'
 else:
  write(f,manifest)
  with zipfile.ZipFile(OUT/'runtime_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
   for p in sorted(set(files)):z.write(p,str(p.relative_to(ROOT)))
 write(OUT/'training_config.json',dict(variants=list(VARIANTS),warm_start_checkpoint_sha256=START_SHA,train_unique=TRAIN_N,validation_unique=VAL_N,test_unique=TEST_N,test_seed_range=[TEST_SEED,TEST_SEED+TEST_N-1],fresh_test_generated_after_selection=True,batch_sources=BATCH,max_epochs=MAX_EPOCHS,min_epochs=MIN_EPOCHS,patience=PATIENCE,min_delta=DELTA,seed=SEED,base_learning_rates=[5e-7,2.5e-6,2.5e-5],frequency_adapter_learning_rate=1e-4,ema_decay=.995,objective='same as previous joint_hann: rectangular direct PSD overlap + amplitude/time/gap regularization + Hann auxiliary weight .15',selection='minimum all-1500 direct joint A/E PSD mismatch; previous model fallback unless validation improves by min_delta',physics='same ideal equal-arm LISA TDI2 A/E, source M 1e5..5e6 Msun, z 1..6, q .3..1, G*h+n; all rows retained',gpu_fraction_per_worker=GPU_FRACTION,minimum_gpu_free_gib=MIN_GPU_FREE/2**30,minimum_ram_free_gib=MIN_RAM_FREE/2**30,shared_cpu_cache_gib=sum(x.stat().st_size for x in (OUT/'cache').glob('*.npy'))/2**30,train_gpu_prefix=4096,validation_gpu_prefix=VAL_N,epoch_checkpoint='model+EMA+optimizer+scheduler+random states; last and best',pilot_weights_reused=False))
 meta=OUT/'launcher.json'
 if meta.exists():
  old=json.loads(meta.read_text());proc=Path('/proc')/str(old['pid'])/'cmdline'
  if proc.exists() and b'prd_frequency.run' in proc.read_bytes():print(json.dumps(old));return
 env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
 with open(OUT/'pipeline.log','ab',buffering=0) as log:
  child=subprocess.Popen(['/home/ljq/miniconda3/envs/gw_ml_env/bin/python','-u','-m','prd_frequency.run'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 v=dict(pid=child.pid,started_unix=time.time(),execution='detached independent coordinator and workers',branch='codex/mask-aware-ringdown-optimization')
 write(meta,v);print(json.dumps(v),flush=True)
if __name__=='__main__':main()
