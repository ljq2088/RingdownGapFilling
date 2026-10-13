import time,json,copy,fcntl,gc,psutil,numpy as np,torch
from prd_physical.common import N,Preprocess,metric,summary,sha,write
from prd_lowrank.common import status,atomic_checkpoint
from .settings import *
from .model import VariableDGF
from .data import Dataset
from .preflight import objective
from .common import check,resource_guard

def opt_for(model):
    base=model.dgf.backbone.dgf.base; decoder=list(base.decoder.parameters()); excluded={id(p) for p in decoder}
    groups=[dict(params=[p for p in base.parameters() if id(p) not in excluded],lr=2e-6),dict(params=decoder,lr=1e-5),dict(params=model.dgf.backbone.dgf.refiner.parameters(),lr=5e-5),dict(params=model.dgf.backbone.joint.parameters(),lr=1e-4),dict(params=model.dgf.frequency.parameters(),lr=5e-5),dict(params=model.adapter.coefficient_adapter.parameters(),lr=2e-4)]
    for g in groups:g['params']=list(g['params'])
    assert len({id(p) for g in groups for p in g['params']})==sum(len(g['params']) for g in groups)
    assert {id(p) for g in groups for p in g['params']}=={id(p) for p in model.parameters() if p.requires_grad}
    return torch.optim.AdamW(groups,weight_decay=1e-4)

def predict(model,data,view):
    model.eval();result=[]
    with torch.inference_mode():
        for lo in range(0,VAL_N,BATCH):
            v=data.batch(np.arange(lo,min(lo+BATCH,VAL_N))+view*VAL_N)
            y=model(v['input'],v['q'],v['mask'],v['bands'],v['q_low']).cpu().numpy()
            assert np.isfinite(y).all();result.append(y)
    return np.concatenate(result)

def validate(model,data):
    cal=np.load(CAL);pre=Preprocess(cal['psd'],'white');h=np.load(CACHE/'validation_truth.npy').astype(float);values={}
    H=np.fft.rfft(h);snr=np.sqrt((4/(2*N)*(abs(H[...,1:-1])**2/cal['psd'][:,1:-1])).sum((1,2)))
    for j,name in enumerate(('variable','fixed66','fixed100')):
        p=pre.decode(predict(model,data,j));mm=metric(h,p,cal['psd'])
        values[name]=dict(psd=summary(mm),time=summary(metric(h,p)),snr_ge10=summary(mm[snr>=10]))
    return dict(score=values['variable']['psd']['mean'],sets=values)

def main():
    path=OUT/'joint_finetune';path.mkdir(exist_ok=True)
    with open(path/'training.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (path/'selected.json').exists():return
        check();assert json.loads((OUT/'prepared.json').read_text())['passed'];resource_guard()
        torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.78);torch.manual_seed(SEED)
        model=VariableDGF().cuda();ema=copy.deepcopy(model).eval()
        for p in ema.parameters():p.requires_grad_(False)
        opt=opt_for(model);sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=15,threshold=DELTA,threshold_mode='abs',min_lr=[2e-8,1e-7,5e-7,1e-6,5e-7,2e-6]);lossfn=objective()
        tr=Dataset('train');va=Dataset('validation');cache_info=dict(validation=va.cache(),training=tr.cache());write(path/'resource_configuration.json',cache_info)
        rng=torch.Generator().manual_seed(SEED+1);history=[];first=1;best_epoch=0;stale=0;best_type='initial'
        if (path/'last.pth').exists():
            c=torch.load(path/'last.pth',map_location='cuda',weights_only=False)
            model.load_state_dict(c['raw']);ema.load_state_dict(c['ema']);opt.load_state_dict(c['optimizer']);sch.load_state_dict(c['scheduler'])
            rng.set_state(c['shuffle_rng'].cpu());torch.set_rng_state(c['torch_rng'].cpu());torch.cuda.set_rng_state(c['cuda_rng'].cpu())
            history=c['history'];first=c['epoch']+1;best=c['best'];best_epoch=c['best_epoch'];stale=c['stale'];best_type=c['best_type']
        else:
            initial=validate(model,va);best=initial['score'];write(path/'initial_validation.json',initial)
            atomic_checkpoint(path/'best.pth',dict(state_dict=model.state_dict(),epoch=0,score=best,weight_type='initial'))
        status(path,'training_started',best=best,best_epoch=best_epoch,exposures=TRAIN_N*VIEWS,cache=cache_info)
        for epoch in range(first,MAX_EPOCHS+1):
            check()
            if psutil.virtual_memory().available<7*2**30:
                tr.ram=[{} for _ in tr.ram];gc.collect()
            resource_guard()
            if torch.cuda.mem_get_info()[0]<4*2**30:
                tr.gpu=[{} for _ in tr.gpu];va.gpu=[{} for _ in va.gpu];torch.cuda.empty_cache()
            while torch.cuda.mem_get_info()[0]<4*2**30:
                status(path,'waiting_for_gpu_headroom',epoch=epoch);time.sleep(10)
                if (OUT/'STOP').exists():return
            if (OUT/'STOP').exists():status(path,'stopped_by_user',epoch=epoch-1);return
            t=time.monotonic();last=0;total=0.;done=0;model.train();order=torch.randperm(TRAIN_N*VIEWS,generator=rng).numpy()
            for lo in range(0,len(order),BATCH):
                v=tr.batch(order[lo:lo+BATCH]);opt.zero_grad(set_to_none=True)
                y=model(v['input'],v['q'],v['mask'],v['bands'],v['q_low']);loss=lossfn(y,v['target'],v['mask'])
                assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step()
                with torch.no_grad():
                    for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,1-EMA)
                    for a,b in zip(ema.buffers(),model.buffers()):a.copy_(b)
                done+=len(y);total+=float(loss.detach())*len(y)
                if time.monotonic()-last>40:
                    elapsed=time.monotonic()-t
                    status(path,'training',epoch=epoch,done=done,total=len(order),best=best,best_epoch=best_epoch,remaining_epoch_seconds=(len(order)-done)*elapsed/done,gpu_peak_mib=torch.cuda.max_memory_allocated()/2**20);last=time.monotonic()
            online=validate(model,va);average=validate(ema,va);typ,result=('ema',average) if average['score']<online['score'] else ('online',online)
            sch.step(result['score'])
            if result['score']<best-DELTA:
                best=result['score'];best_epoch=epoch;best_type=typ;stale=0
                atomic_checkpoint(path/'best.pth',dict(state_dict=(ema if typ=='ema' else model).state_dict(),epoch=epoch,score=best,weight_type=typ))
            else:stale+=1
            row=dict(epoch=epoch,train_loss=total/len(order),validation=result,online_score=online['score'],ema_score=average['score'],best=best,best_epoch=best_epoch,best_type=best_type,stale=stale,epoch_seconds=time.monotonic()-t,lr=[g['lr'] for g in opt.param_groups])
            history.append(row)
            atomic_checkpoint(path/'last.pth',dict(raw=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),shuffle_rng=rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),history=history,epoch=epoch,best=best,best_epoch=best_epoch,best_type=best_type,stale=stale))
            write(path/'history.json',history);median=float(np.median([v['epoch_seconds'] for v in history[-10:]]))
            status(path,'epoch_complete',**row,median_epoch_seconds=median,earliest_remaining_seconds=max(MIN_EPOCHS-epoch,PATIENCE-stale,0)*median,maximum_remaining_seconds=(MAX_EPOCHS-epoch)*median)
            if epoch>=MIN_EPOCHS and stale>=PATIENCE:break
        ck=torch.load(path/'best.pth',map_location='cuda',weights_only=True);model.load_state_dict(ck['state_dict']);final=validate(model,va)
        selected=dict(passed=True,selected_epoch=best_epoch,finished_epoch=epoch,score=best,weight_type=best_type,checkpoint_sha256=sha(path/'best.pth'),validation=final,train_unique=TRAIN_N,observations_per_epoch=TRAIN_N*VIEWS,stop_reason='patience' if stale>=PATIENCE else 'max_epochs',test_read=False)
        write(path/'selected.json',selected);status(path,'completed',**selected)
if __name__=='__main__':main()
