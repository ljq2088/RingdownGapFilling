import copy,gc,json,time,psutil,numpy as np,torch
from prd_physical.common import write,sha
from prd_lowrank.common import status,atomic_checkpoint
from .settings import *
from .model import CoherentDGF,optimizer
from .objective import Objective
from .data import Data
from .validation import validate,eligible
from .common import check,resources,configure

def fit(variant,stage,count,max_epochs,min_epochs,patience,warmstart=None):
    path=OUT/stage/variant;path.mkdir(parents=True,exist_ok=True)
    if (path/'selected.json').exists():return json.loads((path/'selected.json').read_text())
    check();resources();configure();torch.manual_seed(SEED)
    batch=json.loads((OUT/'protocol.json').read_text())['batch']
    model=CoherentDGF(variant).cuda()
    if warmstart is not None:model.load_state_dict(torch.load(warmstart,map_location='cuda',weights_only=True)['state_dict'])
    opt=optimizer(model);base_lr=[g['lr'] for g in opt.param_groups];ema=copy.deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    pairs=[(p,dict(model.named_parameters())[name]) for name,p in ema.named_parameters() if dict(model.named_parameters())[name].requires_grad]
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.7,patience=10,threshold=DELTA,threshold_mode='abs',min_lr=[x*.01 for x in base_lr])
    va=Data('validation',variant);vcache=va.cache();tr=Data('train',variant);tcache=tr.cache(pilot=count<TRAIN_N)
    write(path/'resource_configuration.json',dict(validation=vcache,training=tcache,batch=batch,trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),groups=[dict(name=g['name'],lr=g['lr'],parameters=sum(p.numel() for p in g['params'])) for g in opt.param_groups]))
    baseline=json.loads((OUT/'baseline_validation.json').read_text());rng=torch.Generator().manual_seed(SEED+3);history=[];first=1;stale=0;best_epoch=0;best_type='initial';lossfn=Objective()
    if (path/'last.pth').exists():
        c=torch.load(path/'last.pth',map_location='cuda',weights_only=False)
        assert c['variant']==variant and c['count']==count
        model.load_state_dict(c['raw']);ema.load_state_dict(c['ema']);opt.load_state_dict(c['optimizer']);sch.load_state_dict(c['scheduler'])
        rng.set_state(c['shuffle_rng'].cpu());torch.set_rng_state(c['torch_rng'].cpu());torch.cuda.set_rng_state(c['cuda_rng'].cpu())
        first=c['epoch']+1;history=c['history'];best=c['best'];best_epoch=c['best_epoch'];stale=c['stale'];best_type=c['best_type']
    else:
        initial=validate(model,va,batch);assert eligible(initial,baseline)
        best=initial['score'];write(path/'initial_validation.json',initial)
        atomic_checkpoint(path/'best.pth',dict(state_dict=model.state_dict(),variant=variant,epoch=0,score=best))
    rho=np.load(OUT/'training_snr.npy')[:count];low=np.flatnonzero(rho<20)
    status(OUT,'training_started',stage_name=stage,variant=variant,epoch=first,best=best,training_sources=count,max_epochs=max_epochs,minimum_epochs=min_epochs,early_stop_patience=patience)
    epoch=first-1
    stop_already_reached=epoch>=min_epochs and stale>=patience
    for epoch in ([] if stop_already_reached else range(first,max_epochs+1)):
        check();resources()
        if psutil.virtual_memory().available<8*2**30:tr.ram=[{} for _ in tr.ram];gc.collect()
        if torch.cuda.mem_get_info()[0]<4*2**30:tr.gpu=[{} for _ in tr.gpu];va.gpu=[{} for _ in va.gpu];gc.collect();torch.cuda.empty_cache()
        while torch.cuda.mem_get_info()[0]<4*2**30:
            status(OUT,'waiting_for_gpu',stage_name=stage,variant=variant,epoch=epoch);time.sleep(10);resources()
        if variant=='noise_mix' and stage=='full' and epoch>1 and (epoch-1)%20==0:
            from .prepare import mixed
            del tr;gc.collect();torch.cuda.empty_cache();mixed(TRAIN_N,(epoch-1)//20)
            tr=Data('train',variant);tr.cache()
        t=time.monotonic();last=0.;total=0.;model.training_modes()
        indices=np.concatenate([np.arange(count)+j*TRAIN_N for j in range(3)])
        if variant=='noise_mix' and len(low):
            extra=low[torch.randint(len(low),(int(.3*count),),generator=rng).numpy()]+TRAIN_N*torch.randint(3,(int(.3*count),),generator=rng).numpy()
            indices=np.concatenate((indices,extra))
        order=indices[torch.randperm(len(indices),generator=rng).numpy()]
        for lo in range(0,len(order),batch):
            v=tr.batch(order[lo:lo+batch]);opt.zero_grad(set_to_none=True)
            y=model(v['input'],v['q'],v['mask'],v['bands'],v['q_low']);loss=lossfn(y,v['target'],v['mask'],epoch)
            assert bool(torch.isfinite(loss));loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert bool(torch.isfinite(norm));opt.step();model.clip_constraints()
            with torch.no_grad():torch._foreach_lerp_([a for a,b in pairs],[b for a,b in pairs],1-EMA)
            total+=float(loss.detach())*len(y)
            if time.monotonic()-last>35:
                done=min(lo+batch,len(order));status(OUT,'training',stage_name=stage,variant=variant,epoch=epoch,completed=done,total=len(order),best=best,best_epoch=best_epoch,remaining_epoch_seconds=(len(order)-done)*(time.monotonic()-t)/done);last=time.monotonic()
        with torch.no_grad():
            for a,b in zip(ema.buffers(),model.buffers()):a.copy_(b)
        raw=validate(model,va,batch);avg=validate(ema,va,batch)
        candidates=[(typ,r) for typ,r in [('raw',raw),('ema',avg)] if eligible(r,baseline)]
        typ,result=min(candidates,key=lambda t:t[1]['score']) if candidates else ('ineligible',min((raw,avg),key=lambda r:r['score']))
        sch.step(result['score'])
        if candidates and result['score']<best-DELTA:
            best=result['score'];best_epoch=epoch;best_type=typ;stale=0
            atomic_checkpoint(path/'best.pth',dict(state_dict=(ema if typ=='ema' else model).state_dict(),variant=variant,epoch=epoch,score=best))
        else:stale+=1
        row=dict(epoch=epoch,train_loss=total/len(order),validation_raw=raw,validation_ema=avg,eligible=bool(candidates),best=best,best_epoch=best_epoch,stale=stale,best_type=best_type,epoch_seconds=time.monotonic()-t,observations=len(order),lr=[g['lr'] for g in opt.param_groups])
        history.append(row)
        atomic_checkpoint(path/'last.pth',dict(raw=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),shuffle_rng=rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),history=history,epoch=epoch,best=best,best_epoch=best_epoch,stale=stale,best_type=best_type,variant=variant,count=count))
        write(path/'history.json',history);status(OUT,'epoch_complete',stage_name=stage,variant=variant,epoch=epoch,epoch_seconds=row['epoch_seconds'],best=best,best_epoch=best_epoch,stale=stale,phase_A=result['sets']['variable']['phase_A']['mean'],phase_E=result['sets']['variable']['phase_E']['mean'],eligible=bool(candidates),mean_epoch_seconds=float(np.mean([r['epoch_seconds'] for r in history[-5:]])),minimum_remaining_epochs=max(0,min_epochs-epoch),maximum_remaining_epochs=max_epochs-epoch)
        if epoch>=min_epochs and stale>=patience:break
    ck=torch.load(path/'best.pth',map_location='cuda',weights_only=True);model.load_state_dict(ck['state_dict']);final=validate(model,va,batch);assert eligible(final,baseline)
    selected=dict(variant=variant,stage_name=stage,selected_epoch=best_epoch,finished_epoch=epoch,validation=final,score=best,checkpoint=str(path/'best.pth'),checkpoint_sha256=sha(path/'best.pth'),test_read=False,all_training_sources_retained=True)
    write(path/'selected.json',selected);status(OUT,'stage_completed',**selected)
    del model,ema,opt,va,tr,pairs,lossfn;gc.collect();torch.cuda.empty_cache()
    return selected
