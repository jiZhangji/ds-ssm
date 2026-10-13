# Four-Dataset SAR Results: ig_dsssm_p3_sadd_sd16_b32_seed2_a6000d_v1

All rows use random initialization and report the official test split after validation-set checkpoint selection.

| Dataset | Variant | Placement | P | R | AP50 | AP75 | AP50:95 |
|---|---|---|---:|---:|---:|---:|---:|
| SADD | baseline | baseline | 0.9428 | 0.9527 | 0.9719 | 0.5451 | 0.5432 |
| SADD | full | p3 | 0.9569 | 0.9451 | 0.9692 | 0.5360 | 0.5349 |

## Full-P3 Gain over Baseline

| Dataset | Delta AP50 | Delta AP75 | Delta AP50:95 |
|---|---:|---:|---:|
| SADD | -0.0027 | -0.0091 | -0.0082 |
