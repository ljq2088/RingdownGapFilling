from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FREQUENCY=ROOT/'FillingGap/results_prd_optimization_20261013'
CACHE=FREQUENCY/'cache'
OUT=ROOT/'FillingGap/results_prd_strategy_20261013'
CAL=ROOT/'FillingGap/results_prd_physical_transfer_20261009/calibration_psd.npz'
PRED_TRAIN_N=4096;VAL_N=1500;BATCH=16;GPU_FRACTION=.07
SEED=2026101347
