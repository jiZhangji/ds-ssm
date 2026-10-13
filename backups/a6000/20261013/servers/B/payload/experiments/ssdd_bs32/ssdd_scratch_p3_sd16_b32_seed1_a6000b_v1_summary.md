# Four-Dataset SAR Results: ssdd_scratch_p3_sd16_b32_seed1_a6000b_v1

All rows use random initialization and report the official test split after validation-set checkpoint selection.

| Dataset | Variant | Placement | P | R | AP50 | AP75 | AP50:95 |
|---|---|---|---:|---:|---:|---:|---:|
| SSDD | baseline | baseline | 0.8964 | 0.8004 | 0.8687 | 0.4361 | 0.4553 |
| SSDD | full | p3 | 0.9137 | 0.8059 | 0.8840 | 0.3795 | 0.4490 |

## Full-P3 Gain over Baseline

| Dataset | Delta AP50 | Delta AP75 | Delta AP50:95 |
|---|---:|---:|---:|
| SSDD | +0.0153 | -0.0566 | -0.0063 |
