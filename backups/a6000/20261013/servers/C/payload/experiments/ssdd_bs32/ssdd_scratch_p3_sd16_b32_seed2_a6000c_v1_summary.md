# Four-Dataset SAR Results: ssdd_scratch_p3_sd16_b32_seed2_a6000c_v1

All rows use random initialization and report the official test split after validation-set checkpoint selection.

| Dataset | Variant | Placement | P | R | AP50 | AP75 | AP50:95 |
|---|---|---|---:|---:|---:|---:|---:|
| SSDD | baseline | baseline | 0.9107 | 0.7454 | 0.8549 | 0.3414 | 0.4238 |
| SSDD | full | p3 | 0.8976 | 0.7875 | 0.8515 | 0.2673 | 0.3921 |

## Full-P3 Gain over Baseline

| Dataset | Delta AP50 | Delta AP75 | Delta AP50:95 |
|---|---:|---:|---:|
| SSDD | -0.0034 | -0.0741 | -0.0317 |
