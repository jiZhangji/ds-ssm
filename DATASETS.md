# Dataset Preparation

This repository does not include raw datasets or converted images/labels. Keep them under `datasets/` locally and leave them out of Git.

## Required Datasets

| Dataset | Task | Default local YOLO path | Classes |
|---|---|---|---|
| SSDD / BBox_SSDD | SAR ship detection | `datasets/SSDD_YOLO` | `ship` |
| SAR-AIRcraft-1.0 | SAR aircraft detection | `datasets/SAR_AIRcraft_YOLO` | copied from the COCO category list |

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

