# Four-Dataset SAR Results: sar4_scratch_p3_sd16_v1

All rows use random initialization and report the official test split after validation-set checkpoint selection.

| Dataset | Variant | Placement | P | R | AP50 | AP75 | AP50:95 |
|---|---|---|---:|---:|---:|---:|---:|
| SSDD | baseline | baseline | 0.8958 | 0.7554 | 0.8344 | 0.1939 | 0.3524 |
| SSDD | full | p3 | 0.8823 | 0.8352 | 0.8814 | 0.3376 | 0.4299 |

## Full-P3 Gain over Baseline

| Dataset | Delta AP50 | Delta AP75 | Delta AP50:95 |
|---|---:|---:|---:|
| SSDD | +0.0470 | +0.1436 | +0.0775 |
