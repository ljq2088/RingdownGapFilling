import copy,gc,json,time,numpy as np,torch
from prd_physical.common import write,sha
from prd_lowrank.common import status,atomic_checkpoint
from .settings import *
from .common import guard,check,configure
from .model import TailModel
from .objective import Objective
from .data import Data
from .validation import validate,eligible

def fit(variant,stage,count,max_epochs,min_epochs,patience,warmstart=None):
    path=OUT/stage/variant;path.mkdir(parents=True,exist_ok=True)
    if (path/'selected.json').exists():return json.loads((path/'selected.json').read_text())
    guard();check();configure();torch.manual_seed(SEED)
    model=TailModel(variant).cuda()
    if warmstart:model.load_state_dict(torch.load(warmstart,map_location='cuda',weights_only=True)['state_dict'])
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4 if variant!='qnm_tail' else 1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.7,patience=15,threshold=DELTA,threshold_mode='abs',min_lr=1e-6)
    ema=copy.deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    va=Data('validation',variant);vc=va.cache();tr=Data('train',variant);tc=tr.cache()
    batch=128 if variant=='tail_unet' else 512
    write(path/'resources.json',dict(validation=vc,train=tc,batch=batch,trainable=sum(p.numel() for p in model.parameters())))
    baseline=json.loads((OUT/'baseline_validation.json').read_text());lossfn=Objective();rng=torch.Generator().manual_seed(SEED+1)
    first=1;history=[];stale=0;best_epoch=0;tail_best=baseline['tail']['intermediate']['error']['mean'];best_type='initial'
    if (path/'last.pth').exists():
        ck=torch.load(path/'last.pth',map_location='cuda',weights_only=False)
        assert ck['variant']==variant and ck['count']==count
        model.load_state_dict(ck['raw']);ema.load_state_dict(ck['ema']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        rng.set_state(ck['shuffle_rng'].cpu());torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
        first=ck['epoch']+1;history=ck['history'];best=ck['best'];best_epoch=ck['best_epoch'];best_type=ck['best_type'];stale=ck['stale'];tail_best=ck['tail_best'];del ck
    else:
        initial=validate(model,va);write(path/'initial_validation.json',initial)
        # A pilot checkpoint may trade a little mismatch for much lower tail
        # error. Keep the unchanged baseline as the eligible final candidate.
        best=baseline['score']
        if eligible(initial,baseline) and initial['score']<best:
            best=initial['score'];best_type='warmstart'
            atomic_checkpoint(path/'best.pth',dict(state_dict=model.state_dict(),variant=variant,epoch=0,score=best))
        else:
            origin=TailModel(variant).cuda()
            atomic_checkpoint(path/'best.pth',dict(state_dict=origin.state_dict(),variant=variant,epoch=0,score=best))
            del origin
    rho=np.load(COHERENT/'training_snr.npy')[:count];low=np.flatnonzero(rho<20)
    epoch=first-1
    if not (epoch>=min_epochs and stale>=patience):
        for epoch in range(first,max_epochs+1):
            guard();check();t=time.monotonic();last=0.;total=0.;model.train()
            indices=np.concatenate([np.arange(count)+j*TRAIN_N for j in range(4)])
            if len(low):indices=np.r_[indices,low[torch.randint(len(low),(int(.4*count),),generator=rng).numpy()]+TRAIN_N*torch.randint(4,(int(.4*count),),generator=rng).numpy()]
            order=indices[torch.randperm(len(indices),generator=rng).numpy()]
            for lo in range(0,len(order),batch):
                v=tr.batch(order[lo:lo+batch]);opt.zero_grad(set_to_none=True);p,aux=model(v);loss=lossfn(p,v,aux,epoch)
                assert bool(torch.isfinite(loss)),(variant,epoch,lo);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert bool(torch.isfinite(norm));opt.step();model.clip()
                with torch.no_grad():
                    for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,1-EMA)
                total+=float(loss.detach())*len(p)
                if time.monotonic()-last>35:
                    done=min(lo+batch,len(order));status(OUT,'training',stage_name=stage,variant=variant,epoch=epoch,completed=done,total=len(order),best=best,best_epoch=best_epoch,remaining_epoch_seconds=(len(order)-done)*(time.monotonic()-t)/done);last=time.monotonic()
            raw=validate(model,va);avg=validate(ema,va)
            candidates=[(name,r) for name,r in [('raw',raw),('ema',avg)] if eligible(r,baseline)]
            typ,result=min(candidates,key=lambda x:x[1]['score']) if candidates else ('ineligible',min((raw,avg),key=lambda r:r['score']))
            scheduler.step(result['score'])
            if candidates and result['score']<best-DELTA:
                best=result['score'];best_epoch=epoch;best_type=typ;stale=0
                atomic_checkpoint(path/'best.pth',dict(state_dict=(ema if typ=='ema' else model).state_dict(),variant=variant,epoch=epoch,score=best))
            else:stale+=1
            for name,r,net in [('raw',raw,model),('ema',avg,ema)]:
                tail=r['tail']['intermediate']['error']['mean']
                if r['score']<=baseline['score']*1.02 and all(r[k]['mean']<=baseline[k]['mean']*1.02 for k in ('phase_A','phase_E')) and tail<tail_best:
                    tail_best=tail;atomic_checkpoint(path/'best_tail.pth',dict(state_dict=net.state_dict(),variant=variant,epoch=epoch,score=r['score'],tail=tail_best))
            row=dict(epoch=epoch,train_loss=total/len(order),raw=raw,ema=avg,best=best,best_epoch=best_epoch,best_type=best_type,tail_best=tail_best,stale=stale,epoch_seconds=time.monotonic()-t,observations=len(order),lr=opt.param_groups[0]['lr'])
            history.append(row)
            atomic_checkpoint(path/'last.pth',dict(raw=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),shuffle_rng=rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),epoch=epoch,history=history,best=best,best_epoch=best_epoch,best_type=best_type,stale=stale,tail_best=tail_best,variant=variant,count=count))
            write(path/'history.json',history)
            status(OUT,'epoch_complete',stage_name=stage,variant=variant,epoch=epoch,epoch_seconds=row['epoch_seconds'],best=best,best_epoch=best_epoch,stale=stale,tail_best=tail_best,mean_epoch_seconds=float(np.mean([x['epoch_seconds'] for x in history[-5:]])),phase_A=result['phase_A']['mean'],phase_E=result['phase_E']['mean'],minimum_remaining_epochs=max(0,min_epochs-epoch),maximum_remaining_epochs=max_epochs-epoch)
            if epoch>=min_epochs and stale>=patience:break
    ck=torch.load(path/'best.pth',map_location='cuda',weights_only=True);model.load_state_dict(ck['state_dict']);r=validate(model,va);assert eligible(r,baseline)
    selected=dict(variant=variant,stage_name=stage,selected_epoch=best_epoch,finished_epoch=epoch,score=best,tail_best=tail_best,validation=r,checkpoint=str(path/'best.pth'),checkpoint_sha256=sha(path/'best.pth'),warmstart=str(path/'best_tail.pth') if best_epoch==0 and (path/'best_tail.pth').exists() else str(path/'best.pth'),test_read=False)
    write(path/'selected.json',selected);status(OUT,'stage_completed',**selected)
    del model,ema,opt,scheduler,va,tr,lossfn;gc.collect();torch.cuda.empty_cache()
    return selected
