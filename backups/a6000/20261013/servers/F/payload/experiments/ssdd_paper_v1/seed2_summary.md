# Four-Dataset SAR Results: seed2

All rows use random initialization and report the official test split after validation-set checkpoint selection.

| Dataset | Variant | Placement | P | R | AP50 | AP75 | AP50:95 |
|---|---|---|---:|---:|---:|---:|---:|
| SSDD | baseline | baseline | 0.9152 | 0.8004 | 0.8867 | 0.3828 | 0.4512 |
| SSDD | ds_only | p3 | 0.8931 | 0.7956 | 0.8553 | 0.3497 | 0.4282 |
| SSDD | full | p3 | 0.9029 | 0.8004 | 0.8795 | 0.3472 | 0.4218 |
| SSDD | global | p3 | 0.9252 | 0.7949 | 0.8564 | 0.3252 | 0.4189 |
| SSDD | pfim_only | p3 | 0.9102 | 0.7985 | 0.8727 | 0.3585 | 0.4312 |

## Full-P3 Gain over Baseline

| Dataset | Delta AP50 | Delta AP75 | Delta AP50:95 |
|---|---:|---:|---:|
| SSDD | -0.0072 | -0.0357 | -0.0294 |
