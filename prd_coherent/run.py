import fcntl,json,gc,traceback,time,torch
from prd_physical.common import write,sha
from prd_lowrank.common import status
from .settings import *
from .common import configure,check

def main():
    OUT.mkdir(exist_ok=True)
    with open(OUT/'pipeline.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);configure();check()
        if (OUT/'final_results.json').exists():return
        from .prepare import gram,mixed,snr
        from .train import fit
        from .model import CoherentDGF
        from .data import Data
        from .validation import validate
        snr();gram()
        if not (OUT/'baseline_validation.json').exists():
            model=CoherentDGF('phase_balanced').cuda().eval();data=Data('validation','phase_balanced');data.cache()
            r=validate(model,data,256);assert abs(r['score']-.007923060569647502)<1e-8
            write(OUT/'baseline_validation.json',r);del model,data;gc.collect();torch.cuda.empty_cache()
        pilots=[]
        for variant in VARIANTS:
            if variant=='noise_mix':mixed(PILOT_N)
            pilots.append(fit(variant,'pilot',PILOT_N,PILOT_MAX,PILOT_MIN,PILOT_PATIENCE))
        # Prespecified tie priority favours the hypothesis that requires the
        # larger source population to adapt the inherited feature extractor.
        priority={'encoder_adaptation':0,'data_consistency':1,'noise_mix':2,'phase_balanced':3}
        best=min(pilots,key=lambda r:(r['score'],priority[r['variant']]))
        write(OUT/'pilot_selection.json',dict(selected=best,candidates=pilots,rule='minimum full-validation direct mismatch subject to phase and high-SNR guards'))
        if best['variant']=='noise_mix' and not (OUT/'full/noise_mix/selected.json').exists():
            resume=OUT/'full/noise_mix/last.pth';generation=0
            if resume.exists():
                ck=torch.load(resume,map_location='cpu',weights_only=False)
                generation=int(ck['epoch'])//20;del ck;gc.collect()
            mixed(TRAIN_N,generation)
        selected=fit(best['variant'],'full',TRAIN_N,FULL_MAX,FULL_MIN,FULL_PATIENCE,best['checkpoint'])
        # Completed selection is immutable before the untouched test is read.
        write(OUT/'selection_frozen.json',dict(selection=selected,protocol_sha=sha(OUT/'protocol.json'),frozen_at=time.time(),fresh_test_read=False))
        from .final import main as final
        final()
if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(traceback=traceback.format_exc(),time=time.time()));status(OUT,'failed',error=traceback.format_exc());raise
