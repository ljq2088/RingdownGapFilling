import numpy as np,torch,time,json,gc
from prd_physical.common import write
from prd_lowrank.common import status
from .model import ConditionedDGF,packet,reconstruct
from .data import MixedData
from .settings import *
from .common import check,wait_resources

def main():
    if (OUT/'prepared.json').exists():return
    check();wait_resources();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.78)
    model=ConditionedDGF('noise_phase').cuda().eval()
    for split in ('validation','train'):
        data=MixedData(split);info=data.cache();count=data.count
        for j in range(3):
            path=OUT/'packets'/f'{split}{j}';path.mkdir(parents=True,exist_ok=True)
            arrays={k:np.load(path/(k+'.npy'),mmap_mode='r+') if (path/(k+'.npy')).exists() else np.lib.format.open_memmap(path/(k+'.npy'),mode='w+',dtype='f4',shape=(count,)+shape) for k,shape in PACKET_FIELDS.items()}
            progress=path/'progress.json';done=json.loads(progress.read_text())['completed'] if progress.exists() else 0
            t=time.monotonic();begin=done;last=0
            for lo in range(done,count,128):
                hi=min(lo+128,count);v=data.batch(np.arange(lo,hi)+j*count)
                with torch.inference_mode():
                    p=model.backbone(v['input'],v['q'],v['mask'],v['bands'],v['q_low'])
                    values=packet(v['input'],p,v['mask'],v['q_low'],model.backbone.adapter.basis,model.noise_rms,v['sign'])
                    if lo==0:
                        error=float((reconstruct(model.head,values,model.backbone.adapter.basis)-p).abs().max())
                        assert error<2e-4,error
                for k,a in values.items():
                    a=a.cpu().numpy();assert np.isfinite(a).all();arrays[k][lo:hi]=a
                if time.monotonic()-last>30 or hi==count:
                    for a in arrays.values():a.flush()
                    write(progress,dict(completed=hi,total=count));elapsed=time.monotonic()-t
                    status(OUT,'preparing_frozen_backbone_packets',split=split,view=j,completed=hi,total=count,remaining_view_seconds=(count-hi)*elapsed/max(hi-begin,1));last=time.monotonic();wait_resources()
            del arrays
        del data;gc.collect();torch.cuda.empty_cache()
    write(OUT/'prepared.json',dict(passed=True,train_unique=TRAIN_N,views=3,observations_per_epoch=3*TRAIN_N,third_view='fixed66 fresh independent physical noise, seeds 2158200000..2158259999',basis_rank=128,test_read=False))
if __name__=='__main__':main()
