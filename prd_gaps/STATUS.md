# Implementation and pilot status — 2026-10-18

Branch: `codex/physical-gaps-tdi-imr`, based on `dbda45a`.

The implementation supplies complete-stream masking, IMR signals, propagated
link availability, finite-support whitening features, a new 1,121,938-parameter
network, source-disjoint dataset generation, validation-driven training and
paired availability tests. Source masses span 1e5--5e6 solar masses and redshifts
span 1--6. The pilot contains 128 training and 32 validation sources. No held-out
test set was evaluated.

## Verified numerical properties

- All 14 link-replacement tests pass: six directed links and the simultaneous
  six-link case, for 0.5- and 33-second raw link gaps. Retained samples have zero
  measured replacement error in these tests. The interpolation guard is
  conservative and specific to the equal-arm backend configuration.
- A 33-second gap on link 12 produces 110.5 seconds of unavailable A samples and
  118.5 seconds of unavailable E samples with the current guard. The multi-event
  test yields 189 and 205 seconds respectively.
- Raw missing values, including NaNs, have no influence on model features.
  Accepted FIR features agree with complete-data filtering to 5.95e-8 absolute
  error after float32 conversion.
- Independent-noise RMS matches the calibrated one-sided periodogram through
  Parseval's identity at machine precision.
- The exact marginal noise quadratic averages 76.60 over 256 realizations with
  76 retained degrees of freedom.
- The synthetic low-frequency edge test gives a tapered/hard-gap high-frequency
  power ratio of 2.46e-5. A separate colored-noise covariance calculation gives
  offdiagonal Fourier covariance fractions of 0.051 (complete), 0.211 (hard gap)
  and 0.321 (tapered gap). Tapering changes both spectral leakage and covariance.
- Every Q/K/V projection in all four attention layers has a nonzero gradient.
- Interrupted/resumed optimization agrees with uninterrupted optimization to a
  maximum parameter difference of 2.77e-6 in the CUDA float32 test (tolerance
  1e-5); the best validation scores agree within that tolerance.
- Fixing the source phase at a padding-independent reference frequency and using
  sub-sample peak placement reduces the 2048-versus-3072-second padding mismatch
  to 1.17e-10 and relative time-domain L2 difference to 1.89e-5 for the checked
  source. The four mass/redshift/q boundary cases pass coverage checks.

## Final pilot (v3)

Data: `outputs/physical_gaps_pilot_v3_20261018/`.
Training: `outputs/physical_gaps_pilot_training_v3_20261018/`.
The best validation checkpoint is epoch 15; early stopping occurs at epoch 27.

| Method | Full-record mean mismatch | Ringdown-window mean mismatch | Ringdown median |
| --- | ---: | ---: | ---: |
| Zero-filled noisy input | 0.40116 | 0.39158 | 0.23106 |
| Fixed 0.15-Hz low-pass baseline | 0.27245 | 0.29229 | 0.12798 |
| New network | 0.21459 | 0.27372 | 0.10508 |

These are validation results used for model selection. The network's ringdown
95th-percentile mismatch is 0.99891, with a substantial difficult tail. The
low-pass baseline has a comparable tail. The pilot shows executable learning
and modest aggregate improvement over the simple filter; production recovery
accuracy remains a development target.

The detector-peak-to-six-tau window contains a gap for 16 of the 32 validation
sources. This predefined physical-condition subset exposes the gap-recovery
difficulty that is diluted when the long random-gap records are averaged.

| Mask condition | All 32 mean | 16 sources with ringdown gaps |
| --- | ---: | ---: |
| Simple interval imposed after TDI | 0.22914 | 0.39028 |
| Propagated link availability | 0.27372 | 0.47964 |
| Propagated availability plus 72-s guard on each side | 0.36164 | 0.65579 |

The model was trained on propagated-link masks. The additional guard is a
validation stress condition. Both decreased data coverage and changed mask
geometry contribute to its performance change; the result quantifies sensitivity
to neighboring information. A production ablation should train matched models
for each condition and preserve the same sources and noise realizations.

## Next acceptance gates

1. Expand source diversity and regenerate noise/gaps per training source while
   retaining fixed validation realizations and strict source separation.
2. Select models using ringdown-gap subsets and report full-population values
   alongside them. Measure gap-local amplitude error, retained-noise consistency,
   mass/SNR strata and the fraction of ringdown information removed.
3. Compare raw-only, finite-FIR and taper features with separately trained models;
   scan the FIR length, PSD floor, loss normalization and temporal receptive field.
4. Check a second IMR family and post-merger start-time stability before any new
   conditional-QNM claims. Test source-parameter uncertainty using observed-data
   estimates and a guarded post-merger fitting interval.
5. Freeze the protocol and checkpoint selection, then evaluate an untouched
   independent test set before replacing manuscript results.

The earlier v1/v2 pilot artifacts remain as development records. Their generator
used a start-frequency-dependent phase reference; v3 supersedes them. No v1/v2
result is used in the tables above. Existing manuscript results retain their
original provenance and are unchanged by this work.
