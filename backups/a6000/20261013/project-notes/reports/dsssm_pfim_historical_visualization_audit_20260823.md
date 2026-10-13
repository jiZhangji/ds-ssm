# Historical DS-SSM / PFIM Visualization Audit

## Scope

This is an exploratory visualization audit using recovered SSDD checkpoints. It
does not replace the planned scratch-trained, protocol-matched SAR-Mix ablation.

- Baseline: recovered `mambayolo_t_ssdd_300e_16bs` best checkpoint.
- Method: recovered PFIM ex27 best checkpoint.
- PFIM checkpoint SHA256: `1cebf8735311e5f0461b079d4e50b8b5d957dd6f171532e832821857b240aa09`.
- Execution code: exact recovered `mamba_yolo_ex27-E-new.py`, not the later sweep forward path.
- Images: six SSDD validation images selected by fixed object-count quantiles,
  ranging from one to 24 annotated ships.
- Inference: 640 pixels, FP32, confidence 0.25, NMS IoU 0.7.

## What Was Captured

Forward hooks record the actual tensors used by the historical checkpoint:

- normalized DS-SSM input feature energy;
- local residual `x - x_bar`;
- corrected feature energy after `x_tilde = x - lambda_s (x - x_bar)`;
- token-wise PFIM importance;
- effective scan-drive gate produced by gamma and write modulation.

The learned residual coefficient is `lambda_s = 0.430664`, confirming that the
checkpoint applies a non-trivial local residual correction.

## Observations

The visualization pipeline works and produces spatially aligned feature maps,
but the recovered checkpoint does not support a claim that PFIM always acts as
a foreground mask.

| Image | Objects | PFIM/background importance ratio | Effective-gate ratio | Observation |
|---|---:|---:|---:|---|
| `000531` | 1 | 3.67 | 1.10 | Clear positive example; target receives stronger importance and gate response. |
| `000929` | 1 | 1.04 | 1.01 | Approximately neutral spatial modulation. |
| `000099` | 1 | 0.45 | 0.81 | Counterexample; target-region importance is below background despite higher final confidence. |
| `001081` | 2 | 0.56 | 0.92 | PFIM emphasizes broader scene structure rather than only GT boxes. |
| `001131` | 7 | 0.63 | 0.91 | Local residuals are strong around ships and linear structures, while gating is context-dependent. |
| `001111` | 24 | 0.80 | 0.99 | Dense scene; modulation follows ships and strong land scatterers, not a clean object mask. |

Across these six fixed samples, the mean foreground/background importance ratio
is about `1.19`, but the median is only about `0.72`; the positive mean is driven
mainly by `000531`. Therefore the current evidence supports describing PFIM as a
learned spatial state-update modulator, not as a guaranteed target segmentation
or attention mask.

The local residual and correction magnitude are proportional by construction,
so they must not be presented as two independent visual mechanisms. Likewise,
the historical delta bias is a deterministic affine function of PFIM importance
and should not be used as a separate qualitative proof.

## Recommended Paper Use

The current figures are useful for method debugging and for motivating the new
controlled visualization protocol. They should not yet be used as definitive
paper evidence because the recovered baseline and PFIM checkpoints were not
trained under a fully matched protocol.

For the planned SAR-Mix experiment, generate the same fixed panels for baseline,
DS-only, PFIM-only, and full DS-SSM+PFIM checkpoints. Report the full validation
distribution of foreground/background ratios and include a predeclared positive,
median, and failure case. This will test whether the intended behavior emerges
under controlled random initialization instead of selecting only favorable images.

## Artifacts

- Final figures and tensors: `artifacts/dsssm_motivation_ex27_exact_v2_20260823/`
- Visualization script: `source_code/work_train_single/tools/visualize_dsssm_motivation.py`
- Per-image statistics: `artifacts/dsssm_motivation_ex27_exact_v2_20260823/motivation_summary.csv`
- Reproducibility manifest: `artifacts/dsssm_motivation_ex27_exact_v2_20260823/manifest.json`
