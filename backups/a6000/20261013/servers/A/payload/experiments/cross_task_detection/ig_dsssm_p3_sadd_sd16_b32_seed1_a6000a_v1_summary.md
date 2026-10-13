# Four-Dataset SAR Results: ig_dsssm_p3_sadd_sd16_b32_seed1_a6000a_v1

All rows use random initialization and report the official test split after validation-set checkpoint selection.

| Dataset | Variant | Placement | P | R | AP50 | AP75 | AP50:95 |
|---|---|---|---:|---:|---:|---:|---:|
| SADD | baseline | baseline | 0.9452 | 0.9362 | 0.9694 | 0.5110 | 0.5318 |
| SADD | full | p3 | 0.9395 | 0.9492 | 0.9706 | 0.5300 | 0.5312 |

## Full-P3 Gain over Baseline

| Dataset | Delta AP50 | Delta AP75 | Delta AP50:95 |
|---|---:|---:|---:|
| SADD | +0.0012 | +0.0191 | -0.0006 |
