"""Eight separate smoke samples exercise final reporting; the 6000-source test stays sealed."""
import json, shutil
from pathlib import Path
import torch
from prd_physical.common import write,sha
from .settings import OUT
from .train import train_variant
from . import final

def main():
    sample=OUT/'preflight_run'
    train_variant('repeated_control',sample,128,3,1,100)
    train_variant('noise_augmented',sample,128,3,1,100)
    chosen=[json.loads((sample/k/'selected.json').read_text()) for k in ['repeated_control','noise_augmented']]
    preferred=min(chosen,key=lambda x:x['validation']['score'])['variant']
    write(sample/'selection_frozen.json',dict(candidates=chosen,preferred=preferred,full_training_sources=128,new_test_read=False))
    final.OUT=sample;final.TRAIN_N=128;final.TEST_N=8;final.TEST_SEED=2099100000
    final.main()
    result=json.loads((sample/'final_results.json').read_text());assert result['n']==8
    assert json.loads((sample/'final_metric_audit.json').read_text())['passed']
    assert not (OUT/'fresh_test').exists()
    write(OUT/'final_smoke_gate.json',dict(passed=True,n=8,smoke_seed_range=[2099100000,2099100007],full_benchmark_test_read=False,scope='pipeline smoke data separate from benchmark; 128-source three-epoch weights'))
if __name__=='__main__':main()
