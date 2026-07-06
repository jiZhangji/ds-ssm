# Dataset Preparation

This repository does not include raw datasets or converted images/labels. Keep them under `datasets/` locally and leave them out of Git.

## Required Datasets

| Dataset | Task | Default local YOLO path | Classes |
|---|---|---|---|
| SSDD / BBox_SSDD | SAR ship detection | `datasets/SSDD_YOLO` | `ship` |
| SAR-AIRcraft-1.0 | SAR aircraft detection | `datasets/SAR_AIRcraft_YOLO` | copied from the COCO category list |
| OGSOD | SAR/remote-sensing object detection | `dataset/OGSOD/OGSOD` | `ship` by default |
| ship_dataset_v0 | SAR ship detection | `dataset/yolo/ship_dataset_v0` | `ship` |

## SSDD

The repository contains converters for common SSDD releases:

```powershell
cd source_code/work_train_single
python tools/prepare_ssdd_bbox_voc_to_yolo.py --root D:\path\to\BBox_SSDD\voc_style --out datasets/SSDD_YOLO

# or, for COCO-style SSDD
python tools/prepare_ssdd_coco_to_yolo.py --root D:\path\to\BBox_SSDD\coco_style --out datasets/SSDD_YOLO
```

The same tools are also available under `source_code/work_train_ddp/tools/`.

After conversion, use:

```powershell
python mbyolo_train.py --data ultralytics/cfg/datasets/SSDD_YOLO.yaml --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DS-SSM_Backbone-SSDD-ex7.yaml
```

## SAR-AIRcraft

Convert the COCO-format SAR-AIRcraft release into YOLO format:

```powershell
cd source_code/work_train_single
python tools/convert_sarair_coco2yolo.py --src D:\path\to\SAR_AIRcraft --dst datasets/SAR_AIRcraft_YOLO
```

Expected raw layout:

```text
SAR_AIRcraft/
  annotations/train.json
  annotations/val.json
  annotations/test.json
  images/
```

After conversion, use:

```powershell
python mbyolo_train_sarair.py --data ultralytics/cfg/datasets/SAR_AIRcraft.yaml
```

## Git Policy

Do not commit raw images, labels, cache files, generated training runs, or checkpoints. If a checkpoint must be shared, use a release asset, object storage, or Git LFS rather than normal Git history.

## Hugging Face ZIP Layouts

For the Hugging Face `shimiandeshu/sar-m` ZIPs, first inspect each extracted directory before training:

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/dataset
find OGSOD SAR_AIRcraft SSDD ship_dataset_v0 -maxdepth 3 -type f | sed -n '1,80p'
```

Current server-side detection dataset paths:

```text
/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/dataset/yolo/ship_dataset_v0
/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/dataset/SSDD/SSDD
/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/dataset/SAR_AIRcraft/SAR_AIRcraft
/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/dataset/OGSOD/OGSOD
```

If the raw ZIP folders need to be rebuilt into standard YOLO detection folders under `dataset/det/`, run:

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/ds-ssm/source_code/work_train_single

python tools/prepare_hf_sar_det_datasets.py \
  --dataset-root /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/dataset \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --seed 2026 \
  --force
```

The script auto-detects common layouts: YOLO `images/labels`, flat YOLO image/txt pairs, COCO JSON, and VOC XML.

Use these server-side dataset YAMLs after conversion:

- `ultralytics/cfg/datasets/ship_dataset_v0.yaml`
- `ultralytics/cfg/datasets/OGSOD.yaml`
- `ultralytics/cfg/datasets/SSDD_HF.yaml`
- `ultralytics/cfg/datasets/SAR_AIRcraft_HF.yaml`
