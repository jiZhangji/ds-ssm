# Local Datasets

This directory is intentionally ignored by Git. Put converted YOLO-format datasets here when training locally.

Expected layout:

```text
datasets/
  SSDD_YOLO/
    images/train/
    images/val/
    labels/train/
    labels/val/
  SAR_AIRcraft_YOLO/
    images/train/
    images/val/
    images/test/
    labels/train/
    labels/val/
    labels/test/
```

Dataset YAML templates are stored in both working copies:

- `source_code/work_train_single/ultralytics/cfg/datasets/`
- `source_code/work_train_ddp/ultralytics/cfg/datasets/`

