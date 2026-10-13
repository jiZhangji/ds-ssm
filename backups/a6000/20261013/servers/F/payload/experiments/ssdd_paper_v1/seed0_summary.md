# Four-Dataset SAR Results: seed0

All rows use random initialization and report the official test split after validation-set checkpoint selection.

| Dataset | Variant | Placement | P | R | AP50 | AP75 | AP50:95 |
|---|---|---|---:|---:|---:|---:|---:|
| SSDD | baseline | baseline | 0.8923 | 0.8040 | 0.8718 | 0.3569 | 0.4269 |
| SSDD | ds_only | p3 | 0.9100 | 0.7949 | 0.8720 | 0.4191 | 0.4559 |
| SSDD | full | p3 | 0.9323 | 0.8077 | 0.8809 | 0.3072 | 0.4234 |
| SSDD | global | p3 | 0.9192 | 0.7802 | 0.8683 | 0.4003 | 0.4498 |
| SSDD | pfim_only | p3 | 0.9070 | 0.8391 | 0.9009 | 0.3816 | 0.4482 |

## Full-P3 Gain over Baseline

| Dataset | Delta AP50 | Delta AP75 | Delta AP50:95 |
|---|---:|---:|---:|
| SSDD | +0.0091 | -0.0497 | -0.0035 |
