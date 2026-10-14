from pathlib import Path
import sys,json,numpy as np,torch,inspect
ROOT=Path('/home/ljq/code/RingdownGapFilling_PRD');sys.path.insert(0,str(ROOT))
from prd_coherent.settings import OUT,CACHE,TRAIN_N,VAL_N
from prd_coherent.common import check,configure
from prd_conditioned.data import MixedData
from prd_coherent.model import GapOperator
from prd_physical.common import write,N
from prd_lowrank.common import status
configure();check();report={}
for split in ('train','validation'):
    d=MixedData(split);geoms=[]
    for view in d.views:
        mask=view['mask'].numpy();length=mask.sum(-1).round().astype(int);start=mask.argmax(-1);original=start+(length-66)//2
        assert ((original>=132)&(original<=263)).all()
        assert np.isin(length,np.arange(64,105,2)).all()
        assert ((np.arange(N)[None,:]>=start[:,None])&(np.arange(N)[None,:]<(start+length)[:,None])==mask).all()
        geoms.append(dict(minimum_original_start=int(original.min()),maximum_original_start=int(original.max()),minimum_length=int(length.min()),maximum_length=int(length.max())))
    report[split]=dict(n=d.count,views=geoms)
# Exercise training stage-completion logging's argument contract.
inspect.signature(status).bind(OUT,'stage_completed',variant='phase_balanced',stage_name='pilot',selected_epoch=0)
# Independently check cached Gram matrices against both the operator and its
# adjoint, for the shortest/longest gaps at the boundary of allowed centres.
from prd_coherent.settings import PROBE
B=torch.tensor(np.load(PROBE/'training_direction_basis.npz')['vectors'][:,:128],device='cuda',dtype=torch.float32);op=GapOperator().cuda();saved=np.load(OUT/'mask_gram.npy',mmap_mode='r');error=0.
with torch.inference_mode():
    for index in (0,20,2751,2771):
        origin=132+index//21;length=64+2*(index%21);start=origin-(length-66)//2;m=torch.zeros((1,N),device='cuda');m[:,start:start+length]=1
        f=op(B.T.reshape(1,128,2,N),m).flatten(2)[0];g=f@f.T
        error=max(error,float(abs(g.cpu().numpy()-saved[index]).max()))
        assert bool((torch.linalg.eigvalsh(g+.05*torch.eye(128,device='cuda'))>0).all())
report.update(cached_gram_max_error=error,logging_contract=True,passed=True)
assert error<1e-5
write(OUT/'additional_preflight.json',report);print(json.dumps(report,indent=2))
