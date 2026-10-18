# Kunshan L20 deployment preparation

The training package requires Python 3.10+, NumPy, SciPy and PyTorch.
Precomputed `bank/` contains the calibrated noise PSD, clean source records,
index and checksums. Training from this bank uses the supplied simulator
outputs; LAL/PyCBC/fastlisaresponse are required when generating a new bank.

Use a private environment in the cloud instance. For the inspected CUDA 12.4
driver ceiling, PyTorch provides a CUDA 12.1 build of 2.5.1:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy==1.26.4 scipy==1.13.1
```

This pinned environment is a proposed L20 deployment environment; the local
run uses the existing RTX 5080 environment. Verify the driver and imports on
L20 before running the checks below. The package must be extracted so that
`prd_gaps/` is a direct child of the working directory.

```bash
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m prd_gaps.verify_context --data bank
python -m prd_gaps.check_context_resume --data bank --output smoke_resume
python -m prd_gaps.train_context --data bank --output l20_smoke --kind spectral --precision fp32 --epochs 2 --batch-size 16 --train-limit 256 --val-limit 64 --checkpoint-steps 10 --max-hours 0.5 --pilot
```

After checking finite gradients, actual elapsed time, memory and checkpoint
recovery, launch one bounded experiment:

```bash
python -m prd_gaps.train_context --data bank --output l20_spectral --kind spectral --precision fp32 --epochs 240 --patience 30 --min-delta 0.00005 --batch-size 16 --workers 4 --threads 4 --checkpoint-steps 100 --max-hours 6
```

Resume by adding `--resume` with the same output, batch size, precision and
source limits. Start a fresh output directory for an independent cloud run;
transferring a local optimizer checkpoint across different Torch versions and
GPU architectures requires a fresh equivalence check.

At 3 yuan/hour, the six-hour training allocation is 18 yuan plus environment
setup, verification time and any separately billed storage. Confirm the
platform's current billing and resource availability before submitting a job.
Arrange instance shutdown after the training process exits; stopping Python
alone may leave an instance billed. The prepared commands create no cloud
instance and reserve no paid resources.

Official installation matrix: https://pytorch.org/get-started/previous-versions/
