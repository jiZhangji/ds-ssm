# Four-Dataset SAR Results: corrected_v2_airship_tiles_sd16_b32

All rows use random initialization and report the official test split after validation-set checkpoint selection.

| Dataset | Variant | Placement | P | R | AP50 | AP75 | AP50:95 |
|---|---|---|---:|---:|---:|---:|---:|
| AIR-SARShip-2.0 | baseline | baseline | 0.5654 | 0.4522 | 0.4759 | 0.0746 | 0.1738 |
| AIR-SARShip-2.0 | full | p3 | 0.5854 | 0.4661 | 0.5156 | 0.0909 | 0.1929 |

## Full-P3 Gain over Baseline

| Dataset | Delta AP50 | Delta AP75 | Delta AP50:95 |
|---|---:|---:|---:|
| AIR-SARShip-2.0 | +0.0397 | +0.0162 | +0.0192 |
