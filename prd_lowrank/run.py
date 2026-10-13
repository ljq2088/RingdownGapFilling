import os, json, time, subprocess, fcntl
from pathlib import Path
from prd_physical.common import write
from .settings import *
from .common import status, check_sources

def call(module,*args):
    p=subprocess.run(['/home/ljq/miniconda3/envs/gw_ml_env/bin/python','-u','-m',module,*args],cwd=ROOT)
    if p.returncode:raise RuntimeError(module+' failed: '+str(p.returncode))

def main():
    with open(OUT/'coordinator.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check_sources()
        assert json.loads((OUT/'preflight.json').read_text())['passed']
        if (OUT/'final_results.json').exists():return
        call('prd_lowrank.prepare')
        workers={}
        for kind in VARIANTS:
            path=OUT/kind;path.mkdir(exist_ok=True)
            if (path/'selected.json').exists():continue
            with open(path/'pipeline.log','ab',buffering=0) as log:
                p=subprocess.Popen(['/home/ljq/miniconda3/envs/gw_ml_env/bin/python','-u','-m','prd_lowrank.train',kind],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            workers[kind]=p
        status(OUT,'training_heads',workers={k:p.pid for k,p in workers.items()},device='cpu',train_unique=TRAIN_N,observations_per_epoch=TRAIN_N*VIEWS)
        for kind,p in workers.items():
            code=p.wait()
            if code:
                for other in workers.values():
                    if other.poll() is None:other.terminate()
                raise RuntimeError(kind+' worker failed: '+str(code))
        check_sources()
        selected=[json.loads((OUT/k/'selected.json').read_text()) for k in VARIANTS]
        assert all(x['full_budget'] and x['train_unique']==TRAIN_N for x in selected)
        preferred=min(selected,key=lambda x:x['validation']['score'])['variant']
        frozen=dict(candidates=selected,preferred=preferred,full_training_sources=TRAIN_N,new_test_read=False,test_seed_range=[TEST_SEED,TEST_SEED+TEST_N-1],selection='minimum direct all-1500 validation mismatch; fresh test generated afterwards',backbone_sha=SNAPSHOT_SHA,basis_rank=RANK)
        write(OUT/'selection_frozen.json',frozen)
        call('prd_lowrank.final')

if __name__=='__main__':
    try:main()
    except Exception as e:
        status(OUT,'failed',error=str(e));raise
