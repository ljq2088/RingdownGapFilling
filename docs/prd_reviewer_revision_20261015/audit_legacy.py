import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
R=Path('/home/ljq/code/RingdownGapFilling_PRD');sys.path.insert(0,str(R))
import torch
p=Path('/home/ljq/code/RingdownGapFilling/dataset/train_dataset_signal_data_noise_gaps_HugeRangeParas.pth')
a=torch.load(p,map_location='cpu',weights_only=False)
result=dict(type=str(type(a)),count=len(a),attributes={})
for key,v in vars(a).items():
    result['attributes'][key]=dict(type=str(type(v)),shape=list(v.shape) if hasattr(v,'shape') else None)
Path(__file__).with_name('legacy_dataset_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
