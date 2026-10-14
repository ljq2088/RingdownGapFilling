import json,time,copy,gc,fcntl,numpy as np,torch
from prd_physical.common import write,sha
from prd_lowrank.common import status,atomic_checkpoint
from .model import ResidualHead,reconstruct
from .data import PacketData
from .validation import validate
from .objective import Objective
from .settings import *
from .common import check,wait_resources

def train(variant,folder,count,max_epochs,min_epochs,patience):
    path=OUT/folder;path.mkdir(parents=True,exist_ok=True)
    with open(path/'training.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (path/'selected.json').exists():return json.loads((path/'selected.json').read_text())
        check();wait_resources();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.78);torch.manual_seed(SEED)
        model=ResidualHead(variant).cuda();ema=copy.deepcopy(model).eval()
        for v in ema.parameters():v.requires_grad_(False)
        basis=torch.tensor(np.load(PROBE/'training_direction_basis.npz')['vectors'][:,:RANK],dtype=torch.float32,device='cuda')
        tr=PacketData('train');va=PacketData('validation');lossfn=Objective()
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=3e-4)
        sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=15,threshold=1e-6,threshold_mode='abs',min_lr=2e-6)
        generator=torch.Generator().manual_seed(SEED+1);sources=np.random.default_rng(SEED+2).choice(TRAIN_N,count,replace=False)
        all_indices=np.concatenate([sources+j*TRAIN_N for j in range(3)])
        first=1;history=[];best_epoch=0;stale=0;best_type='initial'
        if (path/'last.pth').exists():
            c=torch.load(path/'last.pth',map_location='cuda',weights_only=False)
            assert c['variant']==variant and c['count']==count
            model.load_state_dict(c['raw']);ema.load_state_dict(c['ema']);opt.load_state_dict(c['optimizer']);sch.load_state_dict(c['scheduler'])
            generator.set_state(c['shuffle_rng'].cpu());torch.set_rng_state(c['torch_rng'].cpu());torch.cuda.set_rng_state(c['cuda_rng'].cpu())
            first=c['epoch']+1;history=c['history'];best=c['best'];best_epoch=c['best_epoch'];stale=c['stale'];best_type=c['best_type']
        else:
            initial=validate(model,va,basis);best=initial['score'];write(path/'initial_validation.json',initial)
            atomic_checkpoint(path/'best.pth',dict(state_dict=model.state_dict(),variant=variant,epoch=0,score=best))
        status(path,'training_started',variant=variant,epoch=first,train_unique=count,observations_per_epoch=len(all_indices),parameters=sum(v.numel() for v in model.parameters()))
        for epoch in range(first,max_epochs+1):
            check();wait_resources();t=time.monotonic();model.train();total=0.;last=0
            if epoch<=5:
                for group in opt.param_groups:group['lr']=3e-4*epoch/5
            order=all_indices[torch.randperm(len(all_indices),generator=generator).numpy()]
            for lo in range(0,len(order),1024):
                v=tr.batch(order[lo:lo+1024]);opt.zero_grad(set_to_none=True);y=reconstruct(model,v,basis);loss=lossfn(y,v['target'],v['mask'])
                assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step()
                with torch.no_grad():
                    for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,.005)
                total+=float(loss.detach())*len(y)
                if time.monotonic()-last>40:
                    status(path,'training',epoch=epoch,completed=min(lo+1024,len(order)),total=len(order),best=best,best_epoch=best_epoch);last=time.monotonic()
            raw=validate(model,va,basis);avg=validate(ema,va,basis);typ,result=('ema',avg) if avg['score']<raw['score'] else ('raw',raw)
            if epoch>5:sch.step(result['score'])
            if result['score']<best-1e-6:
                best=result['score'];best_epoch=epoch;best_type=typ;stale=0
                atomic_checkpoint(path/'best.pth',dict(state_dict=(ema if typ=='ema' else model).state_dict(),variant=variant,epoch=epoch,score=best))
            else:stale+=1
            row=dict(epoch=epoch,train_loss=total/len(order),validation=result,best=best,best_epoch=best_epoch,stale=stale,best_type=best_type,epoch_seconds=time.monotonic()-t,lr=opt.param_groups[0]['lr'])
            history.append(row)
            atomic_checkpoint(path/'last.pth',dict(raw=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),shuffle_rng=generator.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),history=history,epoch=epoch,best=best,best_epoch=best_epoch,stale=stale,best_type=best_type,variant=variant,count=count))
            write(path/'history.json',history);status(path,'epoch_complete',**row)
            if epoch>=min_epochs and stale>=patience:break
        ck=torch.load(path/'best.pth',map_location='cuda',weights_only=True);model.load_state_dict(ck['state_dict']);final=validate(model,va,basis)
        selected=dict(passed=True,variant=variant,best_epoch=best_epoch,finished_epoch=epoch,score=best,validation=final,checkpoint_sha256=sha(path/'best.pth'),train_unique=count,observations_per_epoch=len(all_indices),test_read=False)
        write(path/'selected.json',selected);status(path,'completed',**selected);del tr,va,model,ema,opt;gc.collect();torch.cuda.empty_cache();return selected

def main():
    if not (OUT/'pilot_selection.json').exists():
        results={variant:train(variant,'pilot/'+variant,6000,60,20,20) for variant in VARIANTS}
        # Prefer the full context on an exact numerical tie. All pilot sources come from training.
        variant=min(VARIANTS,key=lambda k:(results[k]['score'],0 if k=='noise_phase' else 1))
        write(OUT/'pilot_selection.json',dict(variant=variant,results=results,rule='minimum full-population variable-gap validation direct mismatch',test_read=False))
    variant=json.loads((OUT/'pilot_selection.json').read_text())['variant']
    train(variant,'head_full',TRAIN_N,600,100,100)
if __name__=='__main__':main()
