# Physical-band CWT adapter continuation, 2026-10-13

This experiment preserves the original DGF double-branch backbone and its current physical A/E residual adapters. It starts both candidates from the frozen 2026-10-10 joint-Hann EMA checkpoint at epoch 798. The learned low-frequency feature adapter has 3,856 parameters and a zero-initialized output head; its initial predictions reproduce the previous checkpoint exactly.

## Empirical motivation

At 2 Hz sampling the original Morlet scales 1 through 16 have kernel centers approximately 0.267 through 1 Hz. Across all 1,500 validation sources, the median fraction of PSD-weighted target energy above 0.2666 Hz is 5.39e-7. The fixed added 16-scale bank has nominal centers 0.002 through 0.25 Hz, matched to the synthetic physical ringdown population. The 528-second record has Fourier resolution about 0.001894 Hz. The lowest CWT scales use the existing reflected boundary extension; their finite-window behavior is part of this experiment's scope.

The exact warm-start checkpoint has zero final-ReLU output for every frequency feature on all 60,000 training sources and all 1,500 validation sources. The maximum pre-ReLU activation is -0.000125708 on training and -0.00191302 on validation. The added path enters after this ReLU; preflight confirms nonzero loss gradients for its output head. These activation observations refer to this checkpoint and this physical population.

## Comparisons and selection

The continuation control and frequency adapter use the same frozen starting model, 60,000 training sources, 1,500 validation sources, physical-space loss, shuffle seed, dropout seed, EMA decay, shared backbone learning rates, batch size and epoch budget. Low-band features depend on the whitened observation and its channel RMS. Training labels enter the objective and validation diagnostics. The feature adapter learns a residual feature mapping into the existing frequency embedding, after its final ReLU and before its fully connected projection. Its output is a learned feature representation. The previous embedding and TimeMixer parameters retain their exact checkpoint values at initialization.

The primary criterion is the arithmetic mean of direct, signed, unmaximized, PSD-weighted joint A/E mismatch across every validation source. Hann overlap, time overlap and low-SNR strata provide secondary diagnostics. The previous checkpoint remains available when validation improvements fall below 1e-6. The training recipe allows 800 epochs, a minimum of 80 epochs and patience of 120 epochs without a qualifying improvement. The pilot uses 12,000 training rows for three epochs; long runs restart from the original frozen checkpoint.

After validation selection freezes, generate 6,000 independent sources with seeds 2032200000 through 2032205999. Compare the selected models and previous checkpoint on the identical fresh test population. Retain every source. Report paired source-bootstrap intervals conditional on the trained checkpoints. The historical synthetic mean 0.00536877 remains a separate numerical benchmark; the physical-population improvement claim uses the same new test set.

## Data, resources and reproducibility

Reuse read-only links to the previous immutable preprocessing caches. Add observed-input low-frequency CWT maps. Each worker uses an allocator limit of 30% of total GPU memory, 4 GiB global free-VRAM guard and 6 GiB available system-RAM guard. Cache the first 4,096 training rows and all validation rows when activation reserve permits. Save last and best weights; resume optimizer, scheduler, EMA and random states after interruption. Code hashes and a source archive freeze at launch.

Run preparation, preflight, pilot and launch with the gw_ml_env Python and PYTHONDONTWRITEBYTECODE=1. A coordinator automatically freezes selection and evaluates the new test set when training finishes. Inspect FillingGap/results_prd_optimization_20261013/status.json and each variant's status.json/history.json. To resume after a WSL restart, run python -m prd_frequency.launch in this worktree with the same runtime.
