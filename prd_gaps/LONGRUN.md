# Peak-overlapping link-gap experiments — 18 October 2026

Branch: `physical-gaps-tdi-imr`. The companion manuscript branch is
`mask-aware-ringdown-optimization`.

## Diagnosis

The v3 pilot has mean direct ringdown-window mismatch 0.47964 on its 16
gap-overlapping validation sources and 0.06781 on the other 16 sources. Three
sources lose over 90% of their ringdown-window time-domain signal energy; their
mean mismatch is 1.04723. All 16 gap-overlapping sources have full-record PSD
SNR above 10. This SNR includes the inspiral and the complete injected signal.
Retained ringdown information requires a separate availability-aware measure.

The short-duration sources have damping times comparable to, or shorter than,
the unavailable TDI interval. The original 128-source pilot also compresses
time by 256 before global attention, giving 128 seconds per token. These
conditions motivate more independent sources, targeted gap augmentation and
finer-resolution context. Their individual causal contributions require matched
ablations.

## Data and training

`source_bank.py` generates 8192 training sources and 512 fixed development
sources. The underlying IMRPhenomTHM population, equal-arm TDI convention and
stationary secondary-noise model follow README.md. Each record has 32768
samples at 2 Hz. Training sources are refreshed with new independent noise and
link outages each epoch. Source parameters supply training targets, mask
placement and evaluation windows; the network receives observed channels,
availability masks and their derived features.

Ninety percent of source indices receive a peak-targeted outage design. The
remaining ten percent receive a uniformly located control outage. Propagation
through the TDI stencil determines actual channel availability and overlap.
One quarter of sources have simultaneous six-link outages; others have a
single directed-link outage. One eighth receive an additional short outage.
The first 15 epochs increase raw link durations from 10--33 to 33--53 seconds.
Thereafter 10% of training views use 100--200-second outages, and an independent
10% use an additional response-sized availability guard. Validation keeps fixed
33--53-second outages, fixed noise and fixed sources throughout selection.

The two new encoders retain one token per 32 samples (16 seconds), with widths
32,64,128,192 and strides 4,2,2,2. Both have a multiscale decoder and learn a
residual around a fixed 0.15-Hz, 129-tap low-pass baseline. One uses four
six-head Transformer layers (2,619,362 parameters). The other uses four global
spectral-convolution blocks, each with 96 latent Fourier modes (29,301,218
parameters). Its spectral operation follows the Fourier-operator construction;
the present application and empirical performance are evaluated here:
https://arxiv.org/abs/2010.08895 .

The loss combines full-record error with ringdown-window time error,
missing-sample error, direct signed PSD overlap and PSD-weighted residual
error. Time and phase remain fixed. Reported validation metrics use float64
NumPy FFTs. The frozen PSD and its median-relative floor follow the pilot
convention. This waveform-distance objective is distinct from the exact
marginal Gaussian likelihood for incomplete observations.

## Selection, recovery and resource limits

The selection metric is mean direct ringdown-window A/E mismatch among
validation sources with actual ringdown gaps. All-source means, the gap median
and 95th percentile, and the removed-energy strata accompany it. The severe
stratum loses more than 90% of unweighted window signal energy. No source is
discarded based on reconstruction quality.

The bounded campaign trains the spectral model, then the attention model, with
fresh weights. Each has a 240-epoch ceiling, a six-hour wall-time ceiling,
batch size 16, four loader workers and four Torch CPU threads. AdamW uses an
initial learning rate 2e-4; a validation plateau reduces it by half. Early
stopping counts 30 unimproved epochs after the curriculum, with minimum
improvement 5e-5. FP32 is used for both experiments.

Atomic checkpoints every 100 optimizer steps and at each epoch store weights,
optimizer, scheduler, sample position, generator state and data identity.
Source/epoch-based augmentation and a sampler generator separate from worker
initialization reproduce the same views and order after restart. The tested
two-epoch interrupt/resume comparison differs by at most 2.28e-7 per parameter.
A completed epoch and a wall-time pause have distinct status values. A power
failure requires restarting the campaign command with `--resume`.

```bash
python -m prd_gaps.campaign --data outputs/physical_gaps_longrun_20261018/bank --output outputs/physical_gaps_longrun_20261018
# After an interruption:
python -m prd_gaps.campaign --data outputs/physical_gaps_longrun_20261018/bank --output outputs/physical_gaps_longrun_20261018 --resume
```

Training reads training and validation splits. An untouched test population,
an alternate waveform family, PSD/window sensitivity and a guarded QNM fit
remain subsequent acceptance checks. Manuscript results retain their existing
provenance during these experiments.

## Precision checks and Kunshan L20

The fixed low-pass convolution is held in FP32 even in mixed precision: red
noise cancellation made BF16 filtering inaccurate. After this correction, the
initial FP32/BF16 output comparison gives mismatch about 7e-9. On trained
64-source development predictions, mean absolute changes in ringdown mismatch
are 7.05e-4 (attention) and 1.08e-3 (spectral), with maxima 0.00447 and 0.00620.
Local batch-16 tests also favor FP32 runtime. These results motivate the FP32
campaign and a separate precision audit on any new accelerator.

Kunshan L20 has 48 GB memory. The platform hardware table lists FP32 throughput
59.35 TFLOPS and FP64 throughput 0.927 TFLOPS. Neural-network training can use
FP32; waveform generation and final metrics can retain CPU float64. A cloud
benchmark determines throughput and suitable batch size. GPU generation and
training timing on the local RTX 5080 do not determine L20 speed.

Sources:
- https://www.scnet.cn/help/docs/mainsite/ai/introduction/index.html
- https://pytorch.org/get-started/previous-versions/

At inspection the Kunshan resource overview showed zero available L20 GPUs;
the creation page quoted 3 yuan per GPU-hour. No cloud job was submitted.
