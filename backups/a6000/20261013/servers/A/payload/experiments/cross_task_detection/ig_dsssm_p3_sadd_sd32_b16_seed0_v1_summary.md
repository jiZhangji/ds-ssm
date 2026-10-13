# Four-Dataset SAR Results: ig_dsssm_p3_sadd_sd32_b16_seed0_v1

All rows use random initialization and report the official test split after validation-set checkpoint selection.

| Dataset | Variant | Placement | P | R | AP50 | AP75 | AP50:95 |
|---|---|---|---:|---:|---:|---:|---:|
| SADD | baseline | baseline | 0.9396 | 0.9377 | 0.9656 | 0.4390 | 0.5021 |
| SADD | full | p3 | 0.9418 | 0.9350 | 0.9631 | 0.4615 | 0.5117 |

## Full-P3 Gain over Baseline

| Dataset | Delta AP50 | Delta AP75 | Delta AP50:95 |
|---|---:|---:|---:|
| SADD | -0.0025 | +0.0225 | +0.0096 |
