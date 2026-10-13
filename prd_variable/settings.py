from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'FillingGap/results_prd_variable_20261013'
from prd_lowrank.settings import CACHE,CAL,PROBE,RANK,TRAIN_N,VAL_N
HEAD=ROOT/'FillingGap/results_prd_lowrank_20261013/noise_augmented/best.pth'
LENGTHS=tuple(range(64,105,2))
SEED=2026101371; NOISE_SEED=2148200000; TEST_SEED=2036200000; TEST_N=6000
VIEWS=2; BATCH=128; MAX_EPOCHS=500; MIN_EPOCHS=100; PATIENCE=100; DELTA=1e-6; EMA=.995
FIELDS={'input':(2,1056),'mask':(1056,),'q':(2,2,16,264),'q_low':(2,2,16,264),'bands':(2,8,1056)}
