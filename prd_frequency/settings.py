from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/"FillingGap/results_prd_physical_transfer_20261009"
PREVIOUS=ROOT/"FillingGap/results_prd_optimization_20261010"
OUT=ROOT/"FillingGap/results_prd_optimization_20261013"
START_CHECKPOINT=PREVIOUS/"joint_hann/best.pth"
START_SHA="7ab7fd7bbead48e0ef5408ecceaaaadacd9e00eeecd0bf9a28185912e3edcea5"
SEED=2026101317
TRAIN_N=60000;VAL_N=1500;TEST_N=6000;TEST_SEED=2032200000
BATCH=128;MAX_EPOCHS=800;MIN_EPOCHS=80;PATIENCE=120;DELTA=1e-6
GPU_FRACTION=.30;MIN_GPU_FREE=4*2**30;MIN_RAM_FREE=6*2**30
KEYS=("input","target","q","mask","bands","q_low")
VARIANTS=("continuation","frequency_adapter")
