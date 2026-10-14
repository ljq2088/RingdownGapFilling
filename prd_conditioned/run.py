import os,sys,time,json,subprocess,fcntl,traceback
from prd_physical.common import write,sha
from prd_lowrank.common import status
from .settings import *
from .common import check

def call(module):
    r=subprocess.run([sys.executable,'-u','-m',module],cwd=ROOT)
    if r.returncode:raise RuntimeError(module+' failed '+str(r.returncode))
def main():
    with open(OUT/'pipeline.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check()
        if (OUT/'final_results.json').exists():return
        call('prd_conditioned.prepare_noise')
        # Protect the earlier run's independent test and GPU resources. Its numerical results do not enter tuning.
        while not (PREVIOUS_OUT/'final_results.json').exists():
            s=json.loads((PREVIOUS_OUT/'status.json').read_text())
            if s.get('stage')=='failed':raise RuntimeError('earlier protected pipeline failed; inspect before proceeding')
            status(OUT,'waiting_for_earlier_pipeline_test',earlier_stage=s.get('stage'));time.sleep(30)
        call('prd_conditioned.prepare');call('prd_conditioned.head_train');call('prd_conditioned.finetune')
        selected=json.loads((OUT/'finetune/selected.json').read_text());check()
        frozen=dict(selection=selected,selection_rule='minimum full-population variable-gap validation direct PSD mismatch, initial baseline included; fixed66/fixed100 secondary',test_seed_range=[TEST_SEED,TEST_SEED+TEST_N-1],new_test_read=False,train_unique=TRAIN_N)
        write(OUT/'selection_frozen.json',frozen);call('prd_conditioned.final')
if __name__=='__main__':
    try:main()
    except Exception:status(OUT,'failed',error=traceback.format_exc());raise
