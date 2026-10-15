# Conditional incident-strain views with noise and frequency contrast

Prepared 2026-10-15 on `codex/mask-aware-ringdown-optimization`.

## Delivery state

Local and cloud manuscripts compiled successfully, 18 pages. Figures 9 and 10 are on pages 13 and 14. The connection recovered after a temporary timeout. Downloaded source archive (52) matched the earlier archive (51), confirming no intervening cloud edits. Uploaded the four revised files and downloaded the final source archive (53) and PDF (33). All four cloud files match the prepared files (with line-ending normalization for text). Cloud build: `1a13f2aa96f-6b778ea796904892`.

Project: https://cn.overleaf.com/project/6ac83816750c2cce452054d2

`overleaf_final.pdf` is the verified cloud manuscript. `cloud_verification.json` records file checks and page count; `overleaf_verified.png` records the new figure in the correct project. Full cloud source ZIP is retained in the Windows output directory. The user confirmed the project identity after checking the open tabs.

## Figures and scientific meaning

Figure 9 retains the original three sources (indices 3874, 4884, 4076), shows the full original noisy A/E records, and adds projected noise to the conditional incident-strain views. Figure 10 adds indices 3322 and 5712, with dominant frequencies 7.298 and 38.670 mHz. Its source panels use the same 0–240 s observer-frame axis. Every plot retains native 0.5-s samples and physical amplitudes.

The source reconstruction fixes the injected detector-frame remnant mass, spin, sky position, inclination, polarization, arrival time and pre-reference waveform history. Eight real coefficients (two phase quadratures of four fundamental modes) are estimated by a PSD-weighted fit to the frozen DGF A/E output. The source synthesis uses the same QNM basis. Equation (17) in the current build defines the linear projection P. The smoothness follows from this conditional waveform prior.

Source-domain noise is s+Pn, using the original realized A/E noise and the same projection P. Gray filling spans a single projected realization. Raw A/E observations occupy the upper two rows. The population evaluation continues to use the frozen direct A/E model outputs.

The lower-frequency example has direct full-record mismatch 0.0002600665 and conditional joint-strain error 0.01238456 over the displayed 240 s. The higher-frequency example has direct mismatch 0.0026314401 and conditional strain error 0.05353739. All test-population statistics and trained checkpoints are unchanged.

## Selection and checks

Screened eight low-frequency and eight high-frequency candidates with rho >= 10, direct mismatch < 0.005 and gap-relative error < 0.15. Low-frequency candidates also cover more than three damping times in the A/E record. Selected two with separated frequencies and conditional strain errors below 6%. Candidate reports include a known-parameter fit directly to the masked observation as a control; these illustrative conditional fits do not establish DGF superiority over that control.

Generated truth reproduces stored float32 targets exactly. The eight basis responses reproduce the clean signal to numerical precision, and the clean-source projection round trips at approximately 1e-15. All fitted bases have rank eight. Native sample arrays, selection report and plotting scripts are archived.

Local TeX Live compilation returned exit code 0. No undefined references, overfull boxes or LaTeX errors were found. REVTeX float-placement warnings remain; all ten figures are present. Figure pages were visually inspected.

## Reproduction

Run `conditional_source_view.py` and `screen_frequency.py` with the WSL gw_ml_env Python from this output directory; these scripts use the frozen result files in the repository and the preceding `ringdown_frequency_examples_20261015` output. Run `plot_frequency_contrast.py` to render the selected pair from candidate arrays. The one-shot manuscript assembly scripts document edits already applied to source/DGF.tex. `stage_compile.py` builds the manuscript with the installed Windows TeX Live.
