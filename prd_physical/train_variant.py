"""Independent white training worker; NEVER evaluates or reads the test set."""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1');os.environ.setdefault('PRD_STATUS_FILE','white/status.json')
import sys,json,traceback
import torch
from .run import ROOT,OUT,train,status,sha
from .resources import GPU_MEMORY_FRACTION

def main():
 if sys.argv[1:]!=['white']:raise ValueError('this worker is reserved for the white variant')
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
 manifest=json.loads((OUT/'runtime_source_manifest.json').read_text())
 for name,expected in manifest.items():
  if sha(ROOT/name)!=expected:raise RuntimeError('runtime source changed: '+name)
 for name in ['prepared.json','metric_gates.json','training_probe.json','resource_change_gates.json']:
  if not json.loads((OUT/name).read_text()).get('passed'):raise RuntimeError('preflight failed: '+name)
 result=train('white');status('variant_completed',variant='white',selection=result)

if __name__=='__main__':
 try:main()
 except BaseException as e:
  if not isinstance(e,SystemExit):status('failed',error=traceback.format_exc())
  raise
