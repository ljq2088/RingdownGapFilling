from pathlib import Path
from prd_conditioned.settings import ROOT, OUT as OLD, CACHE, CAL, PROBE, TRAIN_N, VAL_N
OUT=ROOT/'FillingGap/results_prd_coherent_20261015'
START=OLD/'finetune/best.pth'
SEED=2026101523
VARIANTS=('phase_balanced','encoder_adaptation','data_consistency','noise_mix')
PILOT_N=6000; PILOT_MAX=24; PILOT_MIN=8; PILOT_PATIENCE=8
FULL_MAX=300; FULL_MIN=40; FULL_PATIENCE=40
TEST_SEED=2040200000; TEST_N=6000
DELTA=1e-6; EMA=.995
