# Four-Dataset SAR Results: corrected_v2_sadd_sd32_b32

All rows use random initialization and report the official test split after validation-set checkpoint selection.

| Dataset | Variant | Placement | P | R | AP50 | AP75 | AP50:95 |
|---|---|---|---:|---:|---:|---:|---:|
| SADD | baseline | baseline | 0.9340 | 0.9657 | 0.9833 | 0.5703 | 0.5502 |
| SADD | full | p3 | 0.9306 | 0.9202 | 0.9747 | 0.5055 | 0.5331 |

## Full-P3 Gain over Baseline

| Dataset | Delta AP50 | Delta AP75 | Delta AP50:95 |
|---|---:|---:|---:|
| SADD | -0.0087 | -0.0648 | -0.0171 |
