# SSDD H100 Warmup20 Placement Sweep - Partial Record

## Status

- Sweep ID: `ssdd_1xh100_gpu1_warmup20_v1`
- Record date: 2026-08-17
- Completion: 9 of 16 runs
- Dataset split: 742 train / 186 validation / 232 official test
- Metrics in this record: validation only
- Hardware: one NVIDIA H100 80GB HBM3, physical GPU 1
- Training: 640 pixels, global batch 32, AdamW, lr0 0.0002,
  weight decay 0.0005, cosine LR
- Warmup: 3 optimizer-warmup epochs; 20 DS module-warmup epochs for
  modified models; module warmup is not applicable to the baseline
- Checkpoint evaluated: validation-selected `best.pt`

## Standard Validation Ranking

| Rank | Placement | State dim | P | R | AP50 | AP75 | AP50:95 | Delta vs baseline |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | P2 | 16 | 0.938979 | 0.864019 | 0.915125 | 0.668575 | 0.566725 | +0.018341 |
| 2 | P3 | 8 | 0.958180 | 0.857678 | 0.930708 | 0.630542 | 0.561184 | +0.012800 |
| 3 | P3 | 32 | 0.934557 | 0.878212 | 0.923915 | 0.644184 | 0.560158 | +0.011774 |
| 4 | P4 | 8 | 0.934338 | 0.875079 | 0.920241 | 0.636756 | 0.558172 | +0.009788 |
| 5 | P2 | 32 | 0.956215 | 0.850267 | 0.931906 | 0.641371 | 0.557728 | +0.009344 |
| 6 | P4 | 16 | 0.957294 | 0.839572 | 0.924820 | 0.622704 | 0.549780 | +0.001396 |
| 7 | Baseline | 0 | 0.936287 | 0.874332 | 0.931291 | 0.588958 | 0.548384 | 0.000000 |
| 8 | P3 | 16 | 0.963738 | 0.852731 | 0.928883 | 0.614807 | 0.538396 | -0.009988 |
| 9 | P2 | 8 | 0.960376 | 0.842479 | 0.906385 | 0.576150 | 0.521340 | -0.027044 |

## TTA Validation

| Placement | State dim | Standard AP50:95 | TTA AP50:95 | TTA delta |
|---|---:|---:|---:|---:|
| Baseline | 0 | 0.548384 | 0.516742 | -0.031642 |
| P2 | 8 | 0.521340 | 0.505094 | -0.016246 |
| P2 | 16 | 0.566725 | 0.547302 | -0.019423 |
| P2 | 32 | 0.557728 | 0.540472 | -0.017256 |
| P3 | 8 | 0.561184 | 0.547947 | -0.013237 |
| P3 | 16 | 0.538396 | 0.532700 | -0.005696 |
| P3 | 32 | 0.560158 | 0.548713 | -0.011445 |
| P4 | 8 | 0.558172 | 0.542540 | -0.015632 |
| P4 | 16 | 0.549780 | 0.543021 | -0.006759 |

All nine TTA evaluations are below standard inference. Standard inference
should therefore drive the current validation ranking.

## Interpretation

The current winner is P2 with `state_dim=16`. Its AP50:95 is 0.566725,
which is 0.018341 above the matched baseline. P3 with `state_dim=8` and
P3 with `state_dim=32` are the next strongest completed placements.

P2 is state-dimension sensitive in this partial result: state dimension 16
is best, state dimension 32 remains above baseline, and state dimension 8 is
substantially below baseline. A larger state dimension is not monotonically
better.

These results are not directly comparable with archived SSDD `val/test`
metrics. The current sweep uses a new held-out validation split and trains on
742 rather than all 928 official training images. The official 232-image test
set has not been used for this selection.

## Incomplete Runs

- P4, state dimension 32
- P5, state dimensions 8, 16, and 32
- P2+P3, state dimensions 8, 16, and 32

## Data Quality Note

The evaluator printed GFLOPs as zero and produced a parameter count that did
not match the validator summary. Complexity fields are intentionally omitted
from this record until the evaluator is repaired and the models are profiled
again. This reporting issue does not affect checkpoints or detection metrics.

## Source

User-provided server output in attachment
`dee58d56-875f-45d6-b33e-b901e8b19a14/pasted-text.txt`.
