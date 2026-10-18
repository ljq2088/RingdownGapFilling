"""Compare uninterrupted and within-epoch-resumed optimization across two epochs."""
import argparse,json,subprocess,sys,os
from pathlib import Path
import torch

def run(data,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    common=[sys.executable,'-m','prd_gaps.train_context','--data',data,'--kind','attention','--precision','fp32',
            '--epochs','2','--batch-size','2','--workers','2','--train-limit','8','--val-limit','4','--checkpoint-steps','1','--pilot']
    for name,extra in [('continuous',[]),('resumed',['--max-hours','0.000001']),('resumed',['--resume'])]:
        with (output/(name+'.log')).open('a') as log:
            subprocess.run(common+['--output',str(output/name)]+extra,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
    a=torch.load(output/'continuous/last.pt',map_location='cpu',weights_only=False)
    b=torch.load(output/'resumed/last.pt',map_location='cpu',weights_only=False)
    delta=max(float((a['model'][k]-b['model'][k]).abs().max()) for k in a['model'])
    report=dict(maximum_parameter_difference=delta,best_metric_difference=abs(a['best']-b['best']),
                steps=[a['global_step'],b['global_step']],epochs=[a['epoch'],b['epoch']],tolerance=1e-5)
    assert delta<1e-5 and a['global_step']==b['global_step']==8,report
    (Path(__file__).parent/'reports/context_resume.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.data,a.output)
