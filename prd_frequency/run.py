"""Long-run coordinator: train independently, then freeze before new test generation."""
import os
os.environ.setdefault("OMP_NUM_THREADS","2");os.environ.setdefault("NUMBA_NUM_THREADS","1")
import time,json,subprocess,zipfile,traceback,fcntl
from pathlib import Path
from prd_physical.common import write,sha
from .settings import *
from .train import check_sources
PY="/home/ljq/miniconda3/envs/gw_ml_env/bin/python"
def status(stage,**kw):
 v=dict(stage=stage,pid=os.getpid(),updated_unix=time.time(),**kw);write(OUT/"status.json",v);print(json.dumps(v),flush=True)
def main():
 OUT.mkdir(exist_ok=True)
 with open(OUT/"coordinator.lock","a") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  check_sources()
  if (OUT/"final_results.json").exists():status("completed_previous_run");return
  workers={};env=os.environ.copy();env["PYTHONDONTWRITEBYTECODE"]="1"
  for kind in VARIANTS:
   d=OUT/kind;d.mkdir(exist_ok=True)
   if (d/"selected.json").exists():continue
   metadata=d/"launcher.json"
   if metadata.exists():
    old=json.loads(metadata.read_text());cmd=Path("/proc")/str(old["pid"])/"cmdline"
    if cmd.exists() and b"prd_frequency.train" in cmd.read_bytes():workers[kind]=old["pid"];continue
   with open(d/"pipeline.log","ab",buffering=0) as log:
    child=subprocess.Popen([PY,"-u","-m","prd_frequency.train",kind],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   write(metadata,dict(pid=child.pid,variant=kind,started_unix=time.time()));workers[kind]=child.pid
  status("training_started",workers=workers,test_sealed=True)
  while True:
   if (OUT/"STOP").exists():status("pause_requested_workers_finish_epoch");return
   states={}
   for kind in VARIANTS:
    path=OUT/kind/"status.json"
    states[kind]=json.loads(path.read_text()) if path.exists() else {"stage":"starting"}
    if states[kind]["stage"]=="failed":raise RuntimeError(kind+" worker failed: "+states[kind].get("error",""))
    if kind in workers and not (OUT/kind/"selected.json").exists():
     cmd=Path("/proc")/str(workers[kind])/"cmdline"
     if not cmd.exists() or b"prd_frequency.train" not in cmd.read_bytes():raise RuntimeError(kind+" worker vanished before completing checkpoint selection")
   if all((OUT/k/"selected.json").exists() for k in VARIANTS):break
   status("training",workers=workers,variants={k:{name:v.get(name) for name in ["stage","epoch","sources_done","best_score","best_epoch","stale_epochs","median_epoch_seconds","max_remaining_seconds"]} for k,v in states.items()},test_sealed=True)
   time.sleep(40)
  check_sources();selected=[json.loads((OUT/k/"selected.json").read_text()) for k in VARIANTS]
  baseline=json.loads((OUT/"preflight.json").read_text())["initial_validation"]["score"]
  best=min(selected,key=lambda s:s["best_validation_score"])
  preferred=best["variant"] if best["best_validation_score"]<baseline-DELTA else "previous_joint"
  frozen=dict(preferred=preferred,candidates=selected,previous_validation_score=baseline,selected_unix=time.time(),new_test_read=False,selection="lowest all-1500 direct joint A/E PSD mismatch; keep previous model unless improved by declared min_delta; Hann is secondary only")
  write(OUT/"selection_frozen.json",frozen);status("selection_frozen",preferred=preferred,test_sealed=True)
  with open(OUT/"final_test.log","ab",buffering=0) as log:
   subprocess.run([PY,"-u","-m","prd_frequency.final"],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
  result=json.loads((OUT/"final_results.json").read_text());status("completed",preferred=preferred,n=TEST_N,improved_same_test=result["improved_vs_previous_on_same_new_test"],psd_mean=result["results"][preferred]["psd"]["mean"])
if __name__=="__main__":
 try:main()
 except BaseException as e:
  if not isinstance(e,SystemExit):status("failed",error=traceback.format_exc())
  raise
