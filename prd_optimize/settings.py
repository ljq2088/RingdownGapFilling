from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/"FillingGap/results_prd_physical_transfer_20261009"
OUT=ROOT/"FillingGap/results_prd_optimization_20261010"
PARENT=Path("/home/ljq/code/RingdownGapFilling/FillingGap/results_physical_retraining_20261009")
SEED=2026101017
TRAIN_N=60000;VAL_N=1500;TEST_N=6000;TEST_SEED=2030200000
BATCH=128;MAX_EPOCHS=800;MIN_EPOCHS=80;PATIENCE=120;DELTA=2e-6
GPU_FRACTION=.30;MIN_GPU_FREE=4*2**30;MIN_RAM_FREE=6*2**30
KEYS=("input","target","q","mask","bands")
VARIANTS=("continuation","joint_hann")
