# SSDD Baseline Reproduction Folder

This folder records the preparation work needed to reproduce open-source methods from the MaDiNet SSDD comparison table.

## Tracks

- `yolo`: YOLOv3, YOLOv5n, YOLOv5s, YOLOv8n, YOLOX. These should be reproduced first because they are lightweight, open-source, and close to the current project.
- `mmdetection`: Faster R-CNN, Cascade R-CNN, RetinaNet, FCOS, CenterNet. These can be reproduced using MMDetection/MMYOLO style configs after converting SSDD to COCO format.
- `official repo`: DiffusionDet, ConsistencyDet, DiffDet4SAR, MaDiNet. These require separate Detectron2/DiffusionDet environments.
- `quote only`: methods marked non-open-source in MaDiNet should be cited from the paper unless official code becomes available.

## Required Preparation

1. Freeze the SSDD split used by this project and MaDiNet comparison.
2. Export SSDD labels to both YOLO and COCO formats.
3. Use the same confidence threshold when comparing with MaDiNet table: confidence=0.5.
4. Record Params, GFLOPs, latency, P, R, AP50, and F1 for every reproduced method.
5. Keep each method in an isolated environment or container; do not mix Detectron2 baselines with the current Mamba-YOLO environment.
