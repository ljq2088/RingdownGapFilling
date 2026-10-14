from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from prd_conditioned.settings import CACHE,CAL,OUT as OLD,TRAIN_N,VAL_N
START=OLD/'finetune/best.pth'
OUT=ROOT/'FillingGap/results_prd_tail_20261015'
COHERENT=ROOT/'FillingGap/results_prd_coherent_20261015'
SEED=2026101561
VARIANTS=('tail_basis','tail_unet','qnm_tail')
PILOT_N=12000;PILOT_MAX=60;PILOT_MIN=20;PILOT_PATIENCE=20
FULL_MAX=400;FULL_MIN=80;FULL_PATIENCE=80
TEST_N=6000;TEST_SEED=2042200000
DELTA=1e-6;EMA=.995
