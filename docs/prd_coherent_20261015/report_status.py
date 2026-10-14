from pathlib import Path
import sys,json,time
ROOT=Path('/home/ljq/code/RingdownGapFilling_PRD');sys.path.insert(0,str(ROOT))
from prd_coherent.settings import OUT,OLD
result=dict(current=json.loads((OUT/'status.json').read_text()),now=time.time(),pilots=[])
for p in sorted((OUT/'pilot').glob('*/history.json')):
    rows=json.loads(p.read_text());selected=p.parent/'selected.json'
    best=min([r['best'] for r in rows]);r=rows[-1]
    result['pilots'].append(dict(variant=p.parent.name,epochs=len(rows),selected=selected.exists(),best=best,seconds_per_epoch=sum(x['epoch_seconds'] for x in rows[-5:])/len(rows[-5:]),latest_raw=r['validation_raw']['score'],latest_ema=r['validation_ema']['score']))
for name in ['pilot_selection.json','full/noise_mix/resource_configuration.json','full/noise_mix/selected.json','failure.json']:
    if (OUT/name).exists():
        v=json.loads((OUT/name).read_text())
        if name=='pilot_selection.json':v={k:v['selected'][k] for k in ['variant','score','selected_epoch','finished_epoch']}
        result[name]=v
gen=[]
for line in (OLD/'pipeline.log').read_text(errors='replace').splitlines():
    if not line.startswith('{'):continue
    try:v=json.loads(line)
    except ValueError:continue
    if v.get('stage')=='generating_new_independent_test':gen.append(v)
if len(gen)>1:
    result['previous_test_generation_seconds']=(gen[-1]['updated_unix']-gen[0]['updated_unix'])/(gen[-1]['completed']-gen[0]['completed'])*6000
print(json.dumps(result,indent=2))
