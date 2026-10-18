"""Precision and batch-size timing on trained development checkpoints."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .online import SourceBank
from .context_model import focused_loss
from .train_context import build_model,validation
from torch.utils.data import DataLoader

def run(root):
    torch.set_num_threads(4);root=Path(root);data=SourceBank(root/'bank','val',64)
    loader=DataLoader(data,batch_size=8,num_workers=2)
    result={}
    for kind in ['attention','spectral']:
        state=torch.load(root/('trial_'+kind)/'best.pt',map_location='cpu',weights_only=False)
        model=build_model(kind).cuda();model.load_state_dict(state['model'])
        fp=validation(model,loader,data.psd,'cuda',False);bf=validation(model,loader,data.psd,'cuda',True)
        delta=np.array([a['ringdown_mismatch']-b['ringdown_mismatch'] for a,b in zip(fp['samples'],bf['samples'])])
        row=dict(fp32_gap_mean=fp['gap_mean'],bf16_gap_mean=bf['gap_mean'],mean_absolute_mismatch_change=float(abs(delta).mean()),
                 maximum_absolute_mismatch_change=float(abs(delta).max()),benchmarks=[])
        tensors=[torch.stack([data[i][j] for i in range(16)]) for j in range(4)]
        x,y,v,info=tensors;x,y,v=x.cuda(),y.cuda(),v.cuda();psd=torch.tensor(data.psd,dtype=torch.float32,device='cuda')
        for amp in [False,True]:
            model.train();times=[];torch.cuda.reset_peak_memory_stats()
            for step in range(5):
                model.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):p=model(x)
                loss=focused_loss(p,y,v,info,psd);loss.backward();torch.cuda.synchronize()
                if step>0:times.append(time.perf_counter()-start)
            row['benchmarks'].append(dict(precision='bf16' if amp else 'fp32',batch=16,median_step_seconds=float(np.median(times)),
                                           peak_memory_gib=torch.cuda.max_memory_allocated()/2**30))
        result[kind]=row;print(json.dumps({kind:row}),flush=True)
        del model,x,y,v,p,loss;torch.cuda.empty_cache()
    (Path(__file__).parent/'reports/context_trained_precision.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);run(p.parse_args().root)
