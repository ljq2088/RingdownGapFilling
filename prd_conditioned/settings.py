from pathlib import Path
from prd_variable.settings import ROOT,OUT as PREVIOUS_OUT,CACHE,CAL,PROBE,RANK,TRAIN_N,VAL_N,FIELDS
OUT=ROOT/'FillingGap/results_prd_conditioned_20261014'
START=PREVIOUS_OUT/'joint_finetune/best.pth'
SEED=2026101481; TEST_SEED=2038200000; TEST_N=6000
VARIANTS=('normalized_control','noise_context','noise_phase')
PACKET_FIELDS={'features':(262,),'coefficients':(128,),'scale':(1,),'context':(16,),'phase':(192,),'endpoint_dc':(2,),'endpoint_ny':(2,),'start':(),'length':(),'sign':()}
