# Four-Dataset SAR Results: seed1

All rows use random initialization and report the official test split after validation-set checkpoint selection.

| Dataset | Variant | Placement | P | R | AP50 | AP75 | AP50:95 |
|---|---|---|---:|---:|---:|---:|---:|
| SSDD | baseline | baseline | 0.8790 | 0.7729 | 0.8369 | 0.2817 | 0.3827 |
| SSDD | ds_only | p3 | 0.8936 | 0.7997 | 0.8736 | 0.3584 | 0.4353 |
| SSDD | full | p3 | 0.9221 | 0.7875 | 0.8787 | 0.3686 | 0.4421 |
| SSDD | global | p3 | 0.9084 | 0.7784 | 0.8674 | 0.3804 | 0.4442 |
| SSDD | pfim_only | p3 | 0.8868 | 0.8040 | 0.8733 | 0.3047 | 0.4143 |

## Full-P3 Gain over Baseline

| Dataset | Delta AP50 | Delta AP75 | Delta AP50:95 |
|---|---:|---:|---:|
| SSDD | +0.0417 | +0.0869 | +0.0593 |
