# Lightweight Detection Comparison on MaDiNet-Related SAR Datasets

## Scope and Caution

MaDiNet evaluates three main datasets in its quantitative tables: **SSDD**, **SAR-ShipDataset**, and **SAR-AIRcraft1.0**. The table below prioritizes lightweight or deployable detectors and fills results only when a paper/table explicitly reports the corresponding dataset result. Values are percentages unless otherwise noted.

Important caveat: results from MaDiNet's own tables are internally comparable within that paper. Results from external papers may use different splits, image sizes, confidence thresholds, training recipes, or metric conventions, so they should be used as literature context rather than strict leaderboard comparisons.

## Main Comparison Table

| Method | Type | Source | SSDD AP50 | SSDD AP50:95 | SAR-ShipDataset AP50 | SAR-AIRcraft1.0 AP50 | HRSID AP50 | Params(M) | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| YOLOv3 | YOLO | MaDiNet table | 96.2 | -- | 88.8 | 83.9 | -- | -- | Classic YOLO baseline in MaDiNet. |
| YOLOX | YOLO | MaDiNet table | 96.2 | -- | -- | 81.0 | -- | -- | MaDiNet reports SSDD and SAR-AIRcraft1.0. |
| YOLOv5n | lightweight YOLO | MaDiNet table | 98.0 | -- | 89.3 | 88.2 | -- | -- | Important lightweight baseline in MaDiNet. |
| YOLOv8n | lightweight YOLO | MaDiNet table | 98.1 | -- | -- | 88.2 | -- | -- | Important lightweight baseline in MaDiNet. |
| YOLOv5s | lightweight YOLO | MaDiNet table | 98.3 | -- | -- | -- | -- | -- | Stronger small YOLO baseline on SSDD. |
| DiffDet4SAR | SAR diffusion detector | MaDiNet table | 96.9 | -- | 95.1 | 88.4 | -- | -- | Strong open-source SAR diffusion baseline. |
| MaDiNet | diffusion + Mamba | MaDiNet table | 99.0 | -- | 97.5 | 90.5 | -- | -- | MaDiNet's reported SOTA reference. |
| YOLO-Lite | lightweight YOLOv5-style | External, Remote Sensing 2023 | 94.36 | -- | 92.13 | -- | -- | 7.64 | Reports SSDD and SAR ship dataset; dataset/split may differ from MaDiNet. |
| LH-YOLO | lightweight YOLOv8n-style | External, Remote Sensing 2024 | -- | -- | 93.8 | -- | 96.6 | 1.862 | Reports SAR-ShipDataset and HRSID, not SSDD/SAR-AIRcraft1.0. |
| SHIP-YOLO | lightweight YOLOv8n-style | External via LH-YOLO table | -- | -- | 96.6 | -- | -- | 2.500 | Quoted in LH-YOLO comparison table. |
| YOLOv10n | lightweight YOLO | External, LH-YOLO table | -- | -- | 92.6 | -- | 95.7 | 2.265 | Useful recent nano YOLO reference. |
| CPN-YOLO | YOLOv8-based denoising detector | External, arXiv 2026 | 98.9 | 73.9 / 66.5* | -- | -- | 88.9 | -- | Reports SSDD and HRSID; arXiv text contains both SSDD ablation AP50:95 73.9 and comparison-table AP50:95 66.5, so verify final PDF before citation. |
| YOLO-SAATD | lightweight aircraft YOLO | External, Visual Informatics 2025 | -- | -- | -- | +1~2 over YOLOv8n | -- | -25% vs YOLOv8n | Reports relative gains on SAR-Airport-1.0 and SAR-AIRcraft1.0. |
| Ours: IG-DS-SSM / PFIM ex27 | lightweight Mamba-YOLO | Local measured | 98.704 | 74.022 | -- | -- | -- | 5.9648 | Current best local SSDD result; SAR-ShipDataset/SAR-AIRcraft1.0 not yet evaluated. |

## MaDiNet Original Table Subset

These rows are directly extracted from MaDiNet's reported comparison tables and are therefore the safest to use for a MaDiNet-style baseline table.

| Method | SSDD R | SSDD P | SSDD AP50 | SAR-ShipDataset R | SAR-ShipDataset P | SAR-ShipDataset AP50 | SAR-AIRcraft1.0 AP50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv3 | 94.8 | 95.5 | 96.2 | 89.2 | 84.5 | 88.8 | 83.9 |
| YOLOX | 93.8 | 95.0 | 96.2 | -- | -- | -- | 81.0 |
| YOLOv5n | 95.8 | 97.0 | 98.0 | 90.0 | 85.9 | 89.3 | 88.2 |
| YOLOv8n | 96.4 | 97.3 | 98.1 | -- | -- | -- | 88.2 |
| YOLOv5s | 96.4 | 97.2 | 98.3 | -- | -- | -- | -- |
| DiffDet4SAR | 95.2 | 95.3 | 96.9 | 95.0 | 92.1 | 95.1 | 88.4 |
| MaDiNet | 97.0 | 98.0 | 99.0 | 95.6 | 96.2 | 97.5 | 90.5 |
| Ours: IG-DS-SSM / PFIM ex27 | 95.219 | 97.378 | 98.704 | -- | -- | -- | -- |

## Recommended Paper Usage

For the main paper, use the **MaDiNet Original Table Subset** as the strict comparison frame, because the values come from one experimental protocol. Add the external lightweight methods in a separate "literature lightweight reference" table or appendix with a note about split/metric differences.

If we want to claim cross-dataset robustness, the missing experiments are clear:

1. Run IG-DS-SSM/PFIM on SAR-ShipDataset.
2. Run IG-DS-SSM/PFIM on SAR-AIRcraft1.0 or explain that the current method is ship-focused.
3. Optionally evaluate on HRSID to compare with LH-YOLO and CPN-YOLO.
