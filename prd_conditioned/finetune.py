import time,json,copy,gc,fcntl,psutil,numpy as np,torch
from prd_physical.common import write,sha
from prd_lowrank.common import status,atomic_checkpoint
from .settings import *
from .model import ConditionedDGF
from .data import MixedData
from .validation import validate
from .objective import Objective
from .common import check,wait_resources

def optimizer(model):
    for p in model.parameters():p.requires_grad_(False)
    base=model.backbone.dgf.backbone.dgf.base
    modules=[(model.head,1e-4),(model.backbone.adapter.coefficient_adapter,2e-5),(model.backbone.dgf.frequency,1e-5),(model.backbone.dgf.backbone.joint,5e-6),(model.backbone.dgf.backbone.dgf.refiner,5e-6),(base.decoder,5e-7)]
    groups=[]
    for module,lr in modules:
        params=list(module.parameters())
        for p in params:p.requires_grad_(True)
        groups.append(dict(params=params,lr=lr))
    assert len({id(p) for g in groups for p in g['params']})==sum(len(g['params']) for g in groups)
    return torch.optim.AdamW(groups,weight_decay=3e-4)

def main():
    path=OUT/'finetune';path.mkdir(exist_ok=True)
    with open(path/'training.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (path/'selected.json').exists():return
        check();wait_resources();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.78);torch.manual_seed(SEED)
        variant=json.loads((OUT/'head_full/selected.json').read_text())['variant']
        model=ConditionedDGF(variant).cuda();model.head.load_state_dict(torch.load(OUT/'head_full/best.pth',map_location='cuda',weights_only=True)['state_dict'])
        opt=optimizer(model);ema=copy.deepcopy(model).eval()
        for p in ema.parameters():p.requires_grad_(False)
        sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=8,threshold=1e-6,threshold_mode='abs',min_lr=[1e-6,2e-7,1e-7,5e-8,5e-8,5e-9])
        tr=MixedData('train');va=MixedData('validation');cache=dict(training=tr.cache(),validation=va.cache());write(path/'resource_configuration.json',cache)
        generator=torch.Generator().manual_seed(SEED+3);lossfn=Objective();first=1;history=[];stale=0;best_epoch=0;best_type='initial'
        if (path/'last.pth').exists():
            c=torch.load(path/'last.pth',map_location='cuda',weights_only=False)
            assert c['variant']==variant
            model.load_state_dict(c['raw']);ema.load_state_dict(c['ema']);opt.load_state_dict(c['optimizer']);sch.load_state_dict(c['scheduler'])
            generator.set_state(c['shuffle_rng'].cpu());torch.set_rng_state(c['torch_rng'].cpu());torch.cuda.set_rng_state(c['cuda_rng'].cpu())
            first=c['epoch']+1;history=c['history'];best=c['best'];best_epoch=c['best_epoch'];stale=c['stale'];best_type=c['best_type']
        else:
            initial=validate(model,va,live=True);best=initial['score'];write(path/'initial_validation.json',initial)
            atomic_checkpoint(path/'best.pth',dict(state_dict=model.state_dict(),variant=variant,epoch=0,score=best))
        status(path,'training_started',epoch=first,best=best,best_epoch=best_epoch,observations_per_epoch=3*TRAIN_N,trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),total_parameters=sum(p.numel() for p in model.parameters()))
        for epoch in range(first,201):
            check();wait_resources()
            if psutil.virtual_memory().available<7*2**30:tr.ram=[{} for _ in tr.ram];gc.collect()
            if torch.cuda.mem_get_info()[0]<4*2**30:
                tr.gpu=[{} for _ in tr.gpu];va.gpu=[{} for _ in va.gpu];torch.cuda.empty_cache()
            while torch.cuda.mem_get_info()[0]<4*2**30:
                status(path,'waiting_for_gpu',epoch=epoch);time.sleep(10);wait_resources()
            t=time.monotonic();last=0.;total=0.;model.train()
            # Frozen dropout/normalization stays in deterministic evaluation mode.
            model.backbone.dgf.backbone.dgf.base.encoder.eval()
            order=torch.randperm(3*TRAIN_N,generator=generator).numpy()
            for lo in range(0,len(order),128):
                v=tr.batch(order[lo:lo+128]);opt.zero_grad(set_to_none=True)
                y=model(v['input'],v['q'],v['mask'],v['bands'],v['q_low']);loss=lossfn(y,v['target'],v['mask'])
                assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step()
                with torch.no_grad():
                    for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,.005)
                    for a,b in zip(ema.buffers(),model.buffers()):a.copy_(b)
                total+=float(loss.detach())*len(y)
                if time.monotonic()-last>40:
                    done=min(lo+128,len(order));status(path,'training',epoch=epoch,completed=done,total=len(order),best=best,best_epoch=best_epoch,remaining_epoch_seconds=(len(order)-done)*(time.monotonic()-t)/done);last=time.monotonic()
            raw=validate(model,va,live=True);avg=validate(ema,va,live=True);typ,result=('ema',avg) if avg['score']<raw['score'] else ('raw',raw)
            sch.step(result['score'])
            if result['score']<best-1e-6:
                best=result['score'];best_epoch=epoch;best_type=typ;stale=0
                atomic_checkpoint(path/'best.pth',dict(state_dict=(ema if typ=='ema' else model).state_dict(),variant=variant,epoch=epoch,score=best))
            else:stale+=1
            row=dict(epoch=epoch,train_loss=total/len(order),validation=result,best=best,best_epoch=best_epoch,stale=stale,best_type=best_type,epoch_seconds=time.monotonic()-t,lr=[g['lr'] for g in opt.param_groups])
            history.append(row)
            atomic_checkpoint(path/'last.pth',dict(raw=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),shuffle_rng=generator.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),history=history,epoch=epoch,best=best,best_epoch=best_epoch,stale=stale,best_type=best_type,variant=variant))
            write(path/'history.json',history);status(path,'epoch_complete',**row)
            if epoch>=40 and stale>=50:break
        ck=torch.load(path/'best.pth',map_location='cuda',weights_only=True);model.load_state_dict(ck['state_dict']);final=validate(model,va,live=True)
        selected=dict(passed=True,variant=variant,selected_epoch=best_epoch,finished_epoch=epoch,validation=final,score=best,checkpoint_sha256=sha(path/'best.pth'),test_read=False,all_sources_retained=True)
        write(path/'selected.json',selected);status(path,'completed',**selected)
if __name__=='__main__':main()
