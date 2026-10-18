"""Run two bounded experiments sequentially; resume checkpoints after interruption."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from .train_context import atomic_json

def run(args):
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    data=Path(args.data).resolve();status_path=out/'campaign_status.json'
    if status_path.exists() and not args.resume:raise FileExistsError('Existing campaign: use --resume')
    begun=time.perf_counter();completed=[]
    for kind in ['spectral','attention']:
        destination=out/('long_'+kind)
        if (destination/'status.json').exists():
            previous=json.loads((destination/'status.json').read_text())
            if previous['stage']=='completed':completed.append(kind);continue
        command=[sys.executable,'-u','-m','prd_gaps.train_context','--data',str(data),'--output',str(destination),
                 '--kind',kind,'--precision','fp32','--epochs','240','--patience','30','--min-delta','0.00005',
                 '--batch-size','16','--workers','4','--threads','4','--checkpoint-steps','100','--max-hours','6']
        if (destination/'last.pt').exists():command.append('--resume')
        with (out/('long_'+kind+'.log')).open('a') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            atomic_json(dict(stage='running',controller_pid=os.getpid(),training_pid=process.pid,kind=kind,
                             completed=completed,command=command),status_path)
            code=process.wait()
        if code:
            atomic_json(dict(stage='failed',kind=kind,exit_code=code,completed=completed),status_path)
            raise RuntimeError(f'{kind} exited with code {code}; inspect its log before resuming')
        result=json.loads((destination/'status.json').read_text())
        if result['stage']=='paused' and result.get('reason')=='signal':
            atomic_json(dict(stage='paused',kind=kind,completed=completed),status_path);return
        completed.append(kind)
    atomic_json(dict(stage='finished',experiments=completed,elapsed_seconds=time.perf_counter()-begun),status_path)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--resume',action='store_true');run(p.parse_args())
