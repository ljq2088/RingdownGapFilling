"""Long-context training with deterministic views and within-epoch recovery."""
import argparse, hashlib, json, os, random, signal, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from .online import SourceBank
from .context_model import ContextReconstructor, focused_loss
from .model import GapReconstructor
from .train import window_mismatch
from .metrics import mismatch

def atomic_json(value,path):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(temporary,path)

def atomic_checkpoint(value,path):
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        torch.save(value,handle);handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)

def build_model(kind):
    return GapReconstructor() if kind=='pilot' else ContextReconstructor(kind)

def validation(model,loader,psd,device,amp):
    model.eval();rows=[]
    with torch.no_grad():
        for x,y,valid,info,indices in loader:
            with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=amp):
                pred=model(x.to(device)).float().cpu().numpy()
            target=y.numpy();full=mismatch(pred,target,psd)
            for b,index in enumerate(indices.tolist()):
                start,stop,has_gap,removed=info[b].tolist();start,stop=int(start),int(stop)
                meta=dict(detector_energy_peak_index=start,tau220_estimate_seconds=(stop-start)/12,fs_hz=2.)
                mm=window_mismatch(pred[b],target[b],psd,meta)
                missing=1-valid[b,:,start:stop].numpy()
                norm=float((target[b,:,start:stop].astype(float)**2*missing).sum())
                gap_error=float((((pred[b,:,start:stop]-target[b,:,start:stop]).astype(float)**2)*missing).sum()/max(norm,1e-20))
                rows.append(dict(index=index,full_mismatch=float(full[b]),ringdown_mismatch=mm,
                                 has_gap=bool(has_gap),removed_energy=removed,gap_squared_relative_error=gap_error))
    gap=[r for r in rows if r['has_gap']]
    if not gap:raise ValueError('Validation must include ringdown gaps')
    strata={}
    for name,group in [('severe',[r for r in gap if r['removed_energy']>.9]),('partial',[r for r in gap if r['removed_energy']<=.9])]:
        strata[name+'_count']=len(group)
        strata[name+'_gap_mean']=float(np.mean([r['ringdown_mismatch'] for r in group])) if group else None
    return dict(**strata,full_mean=float(np.mean([r['full_mismatch'] for r in rows])),
                ringdown_mean=float(np.mean([r['ringdown_mismatch'] for r in rows])),
                gap_mean=float(np.mean([r['ringdown_mismatch'] for r in gap])),
                gap_median=float(np.median([r['ringdown_mismatch'] for r in gap])),
                gap_p95=float(np.quantile([r['ringdown_mismatch'] for r in gap],.95)),
                gap_count=len(gap),count=len(rows),samples=rows)

def run(args):
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    amp=args.precision=='bf16' and device=='cuda'
    if amp and not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 unsupported; use --precision fp32')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() and not args.resume:raise FileExistsError('Existing checkpoint: use --resume')
    bank=Path(args.data);manifest=json.loads((bank/'manifest.json').read_text())
    if not manifest.get('complete') and not args.pilot:raise ValueError('Wait for bank completion or use --pilot')
    train=SourceBank(bank,'train',args.train_limit);val=SourceBank(bank,'val',args.val_limit)
    # Hash only immutable protocol fields, so a pilot can run while the bank finishes.
    identity={k:v for k,v in manifest.items() if k not in ('complete','index_sha256')}
    identity['calibration_sha256']=hashlib.sha256((bank/'calibration.npz').read_bytes()).hexdigest()
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    protocol=dict(kind=args.kind,precision=args.precision,batch_size=args.batch_size,seed=args.seed,
                  train_count=len(train),val_count=len(val))
    model=build_model(args.kind).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=6,min_lr=1e-6)
    first=0;next_batch=0;best=float('inf');stale=0;global_step=0;saved_rng=None
    if args.resume and (out/'last.pt').exists():
        state=torch.load(out/'last.pt',map_location=device,weights_only=False)
        if state['manifest_hash']!=fingerprint or state.get('protocol')!=protocol:raise ValueError('Resume protocol or architecture mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        first,next_batch,best,stale,global_step=state['epoch'],state['next_batch'],state['best'],state['stale'],state['global_step']
        saved_rng=state
    elif args.init_checkpoint:
        state=torch.load(args.init_checkpoint,map_location=device,weights_only=False)
        if state.get('kind')!=args.kind:raise ValueError('Initialization architecture mismatch')
        model.load_state_dict(state['model'])
    config=dict(vars(args),device=device,parameters=sum(p.numel() for p in model.parameters()),
                manifest_hash=fingerprint,training_sources=len(train),validation_sources=len(val),
                selection_metric='mean direct PSD mismatch on validation records with ringdown gaps',
                source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    atomic_json(config,out/'config.json')
    psd=torch.from_numpy(train.psd.astype(np.float32)).to(device)
    loader_generator=torch.Generator()
    # Keep sample order independent of worker initialization and persistence.
    loader=DataLoader(train,batch_size=args.batch_size,sampler=RandomSampler(train,generator=loader_generator),num_workers=args.workers,
                      persistent_workers=args.workers>0,pin_memory=device=='cuda',generator=torch.Generator().manual_seed(712))
    val_loader=DataLoader(val,batch_size=args.batch_size,shuffle=False,num_workers=min(args.workers,2),
                          persistent_workers=args.workers>0,pin_memory=device=='cuda',generator=torch.Generator().manual_seed(711))
    stop_requested=[False]
    def request_stop(signum,frame):stop_requested[0]=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    begun=time.perf_counter();epoch_times=[];step_times=[]
    def checkpoint(epoch,batch):
        return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                    epoch=epoch,next_batch=batch,best=best,stale=stale,global_step=global_step,kind=args.kind,
                    manifest_hash=fingerprint,protocol=protocol,torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else [])
    def status(stage,epoch,batch,**extra):
        average=float(np.median(step_times[-50:])) if step_times else None
        value=dict(stage=stage,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),
                   epoch=epoch+1,batch=batch,batches_per_epoch=len(loader),global_step=global_step,
                   best_gap_mismatch=best if np.isfinite(best) else None,stale_epochs=stale,
                   recent_step_seconds=average,elapsed_seconds=time.perf_counter()-begun,**extra)
        if epoch_times:
            value['mean_epoch_seconds']=float(np.mean(epoch_times[-3:]))
            value['remaining_epoch_budget_seconds']=value['mean_epoch_seconds']*max(0,args.epochs-epoch-1)
        atomic_json(value,out/'status.json')
    status('starting',first,next_batch)
    if first>=args.epochs:raise ValueError('Epoch budget is already exhausted')
    for epoch in range(first,args.epochs):
        train.set_epoch(epoch);loader_generator.manual_seed(args.seed+epoch)
        model.train();loss_total=0.;seen=0;epoch_start=time.perf_counter();last_step=epoch_start
        for batch,(x,y,valid,info,indices) in enumerate(loader):
            if epoch==first and batch<next_batch:continue
            if saved_rng is not None:
                torch.set_rng_state(saved_rng['torch_rng'].cpu())
                if device=='cuda':torch.cuda.set_rng_state_all([s.cpu() for s in saved_rng['cuda_rng']])
                saved_rng=None
            x,y,valid=x.to(device,non_blocking=True),y.to(device,non_blocking=True),valid.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=amp):prediction=model(x)
            loss=focused_loss(prediction,y,valid,info,psd)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss; prior checkpoint retained')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(grad):raise FloatingPointError('Nonfinite gradients; prior checkpoint retained')
            optimizer.step();global_step+=1
            loss_total+=float(loss.detach())*len(indices);seen+=len(indices)
            now=time.perf_counter();step_times.append(now-last_step);last_step=now
            if global_step%10==0:status('training',epoch,batch+1,train_loss=loss_total/max(seen,1))
            stop_time=now-begun>=args.max_hours*3600
            if global_step%args.checkpoint_steps==0 or stop_requested[0] or stop_time:
                atomic_checkpoint(checkpoint(epoch,batch+1),out/'last.pt')
            if stop_requested[0] or stop_time:
                status('paused',epoch,batch+1,reason='signal' if stop_requested[0] else 'wall-time budget')
                print(json.dumps(dict(stage='paused',epoch=epoch+1,next_batch=batch+1)),flush=True);return
        next_batch=0;status('validating',epoch,len(loader))
        metrics=validation(model,val_loader,train.psd,device,amp)
        score=metrics['gap_mean'];scheduler.step(score)
        # Curriculum finishes at epoch 15. Early stopping begins afterwards.
        improved=score<best-args.min_delta
        if improved:best=score;stale=0
        elif epoch>=15:stale+=1
        elapsed=time.perf_counter()-epoch_start;epoch_times.append(elapsed)
        record={k:v for k,v in metrics.items() if k!='samples'}
        record.update(epoch=epoch+1,train_loss=loss_total/max(seen,1),seconds=elapsed,best=best,
                      stale=stale,learning_rate=optimizer.param_groups[0]['lr'])
        with (out/'history.jsonl').open('a') as handle:handle.write(json.dumps(record)+'\n')
        state=checkpoint(epoch+1,0);atomic_checkpoint(state,out/'last.pt')
        if improved:
            atomic_checkpoint(state,out/'best.pt');atomic_json(metrics,out/'best_validation.json')
        status('epoch_complete',epoch,len(loader),metrics=record)
        print(json.dumps(record),flush=True)
        if stale>=args.patience:
            status('completed',epoch,len(loader),reason='early stopping');return
    status('completed',args.epochs-1,len(loader),reason='epoch budget')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--kind',choices=['attention','spectral','pilot'],default='attention')
    p.add_argument('--precision',choices=['fp32','bf16'],default='bf16')
    p.add_argument('--epochs',type=int,default=160);p.add_argument('--patience',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--workers',type=int,default=4);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--seed',type=int,default=743);p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--min-delta',type=float,default=1e-4)
    p.add_argument('--checkpoint-steps',type=int,default=200);p.add_argument('--max-hours',type=float,default=12.)
    p.add_argument('--train-limit',type=int);p.add_argument('--val-limit',type=int)
    p.add_argument('--resume',action='store_true');p.add_argument('--pilot',action='store_true');p.add_argument('--init-checkpoint')
    run(p.parse_args())
