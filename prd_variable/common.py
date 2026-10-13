import json,psutil,time
from prd_physical.common import sha,write
from prd_lowrank.common import status,atomic_checkpoint
from .settings import ROOT,OUT

def check():
    for name,expected in json.loads((OUT/'runtime_source_manifest.json').read_text()).items():
        if sha(ROOT/name)!=expected:raise RuntimeError('frozen source changed: '+name)
    for name,expected in json.loads((OUT/'asset_manifest.json').read_text()).items():
        if sha(ROOT/name)!=expected:raise RuntimeError('frozen asset changed: '+name)

def resource_guard():
    while psutil.virtual_memory().available<6*2**30:
        status(OUT,'waiting_for_ram',available_gib=psutil.virtual_memory().available/2**30);time.sleep(10)
        if (OUT/'STOP').exists():raise RuntimeError('STOP requested')
