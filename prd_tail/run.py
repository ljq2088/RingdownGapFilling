import fcntl,gc,json,time,traceback,torch
from prd_physical.common import write,sha
from prd_lowrank.common import status
from .settings import *
from .common import configure,check,guard

def main():
    OUT.mkdir(exist_ok=True)
    with open(OUT/'pipeline.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);configure();guard();check()
        if (OUT/'final_results.json').exists():return
        from .prepare import main as prepare
        prepare()
        if not (OUT/'preflight.json').exists():
            from .preflight import main as preflight
            preflight()
        from .train import fit
        if not (OUT/'fit_smoke.json').exists():
            smoke=[fit(v,'smoke_fit',64,1,1,1) for v in VARIANTS]
            write(OUT/'fit_smoke.json',dict(passed=True,variants=[x['variant'] for x in smoke],complete_epoch_checkpoint_selection=True,test_read=False))
        pilots=[fit(v,'pilot',PILOT_N,PILOT_MAX,PILOT_MIN,PILOT_PATIENCE) for v in VARIANTS]
        priority={'qnm_tail':0,'tail_unet':1,'tail_basis':2}
        ordered=sorted(pilots,key=lambda x:(x['score'],x['tail_best'],priority[x['variant']]))
        write(OUT/'pilot_selection.json',dict(full_candidates=ordered[:2],all_candidates=pilots,rule='Two lowest full-validation direct mismatches; tail error breaks ties. No test data used.'))
        completed=[fit(x['variant'],'full',TRAIN_N,FULL_MAX,FULL_MIN,FULL_PATIENCE,x['warmstart']) for x in ordered[:2]]
        best=min(completed,key=lambda x:(x['score'],x['tail_best']))
        baseline=json.loads((OUT/'baseline_validation.json').read_text())
        write(OUT/'selection_frozen.json',dict(selection=best,all_full=completed,protocol_sha256=sha(OUT/'protocol.json'),frozen_at=time.time(),fresh_test_read=False))
        if best['score']>=baseline['score']-DELTA:
            result=dict(passed=False,reason='No validation-controlled direct-mismatch improvement',selection=best,test_read=False,existing_paper_results_unchanged=True)
            write(OUT/'final_results.json',result);status(OUT,'completed_without_validation_gain',best=best['score']);return
        from .final import main as final
        final()

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(traceback=traceback.format_exc(),time=time.time()));status(OUT,'failed',error=traceback.format_exc());raise
