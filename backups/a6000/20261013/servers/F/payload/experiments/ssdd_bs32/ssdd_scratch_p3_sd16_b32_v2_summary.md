# Four-Dataset SAR Results: ssdd_scratch_p3_sd16_b32_v2

All rows use random initialization and report the official test split after validation-set checkpoint selection.

| Dataset | Variant | Placement | P | R | AP50 | AP75 | AP50:95 |
|---|---|---|---:|---:|---:|---:|---:|
| SSDD | baseline | baseline | 0.8854 | 0.8207 | 0.8863 | 0.4250 | 0.4568 |
| SSDD | full | p3 | 0.8992 | 0.8040 | 0.8848 | 0.3244 | 0.4249 |

## Full-P3 Gain over Baseline

| Dataset | Delta AP50 | Delta AP75 | Delta AP50:95 |
|---|---:|---:|---:|
| SSDD | -0.0015 | -0.1006 | -0.0320 |
