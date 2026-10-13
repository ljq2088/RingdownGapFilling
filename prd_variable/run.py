import sys,json,subprocess,fcntl,traceback
from prd_physical.common import write
from prd_lowrank.common import status
from .settings import *
from .common import check

def call(module):
    r=subprocess.run([sys.executable,'-u','-m',module],cwd=ROOT)
    if r.returncode:raise RuntimeError(module+' failed '+str(r.returncode))
def main():
    with open(OUT/'pipeline.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check()
        assert json.loads((OUT/'preflight.json').read_text())['passed']
        if (OUT/'final_results.json').exists():return
        call('prd_variable.prepare');call('prd_variable.train')
        selected=json.loads((OUT/'joint_finetune/selected.json').read_text());check()
        frozen=dict(selection=selected,selection_rule='minimum full-population uniform-variable-gap validation direct PSD mismatch; raw/EMA checkpoint candidates; fixed66/fixed100 secondary',new_test_read=False,test_seed_range=[TEST_SEED,TEST_SEED+TEST_N-1],full_training_sources=TRAIN_N)
        write(OUT/'selection_frozen.json',frozen);call('prd_variable.final')
if __name__=='__main__':
    try:main()
    except Exception:status(OUT,'failed',error=traceback.format_exc());raise
