from pathlib import Path
import sys,json,time,os
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path('/home/ljq/code/RingdownGapFilling_PRD');sys.path.insert(0,str(ROOT))
from prd_tail.settings import OUT,PILOT_MAX,PILOT_MIN,FULL_MIN,FULL_MAX
status=json.loads((OUT/'status.json').read_text());pid=status.get('pid');alive=False
if pid:
    try:os.kill(pid,0);alive=True
    except ProcessLookupError:pass
result=dict(now_beijing=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),pid_alive=alive,status=status,stages=[])
for path in sorted(OUT.glob('*/*/history.json')):
    rows=json.loads(path.read_text());last=rows[-1];best=min(rows,key=lambda r:r['best']);parent=path.parent
    eligible=[]
    for row in rows:
        for name in ('raw','ema'):
            r=row[name]
            if abs(r['score']-last['best'])<1e-10:eligible.append((row['epoch'],name,r))
    record=dict(stage=parent.parent.name,variant=parent.name,epochs=len(rows),best=last['best'],best_epoch=last['best_epoch'],tail_best=last['tail_best'],recent_epoch_seconds=sum(r['epoch_seconds'] for r in rows[-5:])/len(rows[-5:]),last_raw=last['raw']['score'],last_ema=last['ema']['score'])
    if eligible:
        epoch,name,r=eligible[-1];record['selected_diagnostics']=dict(epoch=epoch,weights=name,phase_A=r['phase_A']['mean'],phase_E=r['phase_E']['mean'],tail=r['tail'])
    result['stages'].append(record)
for name in ('pilot_selection.json','selection_frozen.json','final_results.json','failure.json'):
    if (OUT/name).exists():result[name]=json.loads((OUT/name).read_text())
print(json.dumps(result,indent=2))
