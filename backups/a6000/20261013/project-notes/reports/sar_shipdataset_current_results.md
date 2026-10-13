# SAR-ShipDataset Current New-Server Results

Updated: 2026-07-23

## Dataset Protocol

- Dataset directory: `dataset/yolo/ship_dataset_v0`
- Train: 31,783 images
- Validation: 3,972 images
- Test: 3,974 images and 5,123 ship instances
- Input size used by the recorded evaluations: 512

The total of 39,729 images matches the revised SAR-ShipDataset size described by MaDiNet. The local train/validation/test split was generated with the project preparation script and a fixed random seed. Exact sample-level equivalence with MaDiNet's unpublished split has not been established.

## Test Results

| Method | Scale | P | R | F1 | AP50 | AP50:95 | Status |
|---|---|---:|---:|---:|---:|---:|---|
| IG-DS-SSM/PFIM | T | 0.907854 | 0.889908 | 0.898790 | 0.948440 | 0.578182 | Completed evaluation |
| IG-DS-SSM/PFIM | S | 0.922367 | 0.915870 | 0.919107 | 0.956093 | 0.531767 | Completed evaluation |
| IG-DS-SSM/PFIM + TTA | S | 0.915556 | 0.905808 | 0.910650 | 0.952300 | 0.535373 | Completed evaluation |
| Mamba-YOLO | T | 0.914006 | 0.890045 | 0.901866 | 0.946158 | 0.495684 | 300 epochs, batch changed during training |
| Mamba-YOLO | B | 0.921204 | 0.912820 | 0.916993 | 0.953700 | 0.515411 | Interim checkpoint around epoch 159 |

## Experiment Paths

| Result | New-server run directory |
|---|---|
| IG-DS-SSM/PFIM T | `output_dir/ship_dataset_v0/dsssm_pfim_2xh200_bs384_20260717_154755` |
| IG-DS-SSM/PFIM S | `output_dir/ship_dataset_v0/ship_dsssm_s_from_epoch20_nowarmup_2xh200_bs256_20260718_191329` |
| Mamba-YOLO T | `output_dir/ship_dataset_v0_baselines/ship_original_mambayolo_t_2xh200_bs256_20260719_172747` |
| Mamba-YOLO B | `output_dir/ship_dataset_v0_baselines/ship_original_mambayolo_s_matched_b_2xh200_bs256_20260719_172747` |

## Interpretation

1. The lightweight T-scale IG-DS-SSM/PFIM model improves AP50 by 0.23 percentage points and AP50:95 by 8.25 points over the archived T baseline evaluation. Precision is 0.62 points lower and recall is nearly unchanged.
2. This large AP50:95 gain is promising but not yet a controlled module ablation. The baseline used global batch 256 for epochs 1-198 and batch 128 for epochs 199-300. The method run used a different large-batch schedule.
3. The S-scale method improves AP50 by 0.24 points and AP50:95 by 1.64 points over the available B-scale reference, but the reference was evaluated before 300 epochs. This row is provisional.
4. TTA is not beneficial for the S-scale model: AP50 decreases by 0.38 points and AP50:95 increases by only 0.36 points while precision and recall both decline.
5. Against MaDiNet's reported SAR-ShipDataset AP50 of 97.5, the T-scale method is 2.66 points lower and the S-scale method is 1.89 points lower. Since exact split equivalence is unverified, this is literature context rather than a strict leaderboard comparison.

## Publication Rule

Use the T-scale method as the main lightweight cross-dataset result. Keep the S-scale result as a scale study. Label all Mamba-YOLO baseline deltas as archive comparisons until matched 300-epoch baselines are trained with identical batch, optimizer, augmentation, seed, and initialization.

