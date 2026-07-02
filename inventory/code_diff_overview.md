# Code Diff Overview

## Summary

- Compared tracked priority code/config hashes across `229` paths.
- Identical tracked priority files across all three servers: `224`.
- Differing or missing tracked priority files: `5`.
- Scope: tracked `*.py`, `*.yaml`, `*.yml`, and priority folders from Git plus visible untracked Git status. This is a metadata/hash inventory, not a full code download.

## Differing Tracked Files

| File | 1080Ti | 3090-A | 3090-B |
|---|---|---|---|
| `mbyolo_train.py` | `43a95c335b55` | `3427e9cdd3b9` | `3427e9cdd3b9` |
| `selective_scan/setup.py` | `36019920897a` | `cb67db4a7828` | `cb67db4a7828` |
| `ultralytics/engine/trainer.py` | `33b19dade2e4` | `024587c50417` | `024587c50417` |
| `ultralytics/nn/tasks.py` | `e1c96ad68a32` | `c22332f2942c` | `c22332f2942c` |
| `ultralytics/utils/metrics.py` | `015cd5b295d7` | `015cd5b295d7` | `89480260a543` |

## Git Status Presence Matrix

| Status line | 1080Ti | 3090-A | 3090-B |
|---|---:|---:|---:|
| ` M mbyolo_train.py` | yes | yes | yes |
| ` M selective_scan/setup.py` | yes |  |  |
| ` M ultralytics/cfg/default.yaml` | yes | yes | yes |
| ` M ultralytics/engine/trainer.py` | yes |  |  |
| ` M ultralytics/nn/modules/__init__.py` | yes | yes | yes |
| ` M ultralytics/nn/modules/mamba_yolo.py` | yes | yes | yes |
| ` M ultralytics/nn/tasks.py` | yes | yes | yes |
| ` M ultralytics/utils/metrics.py` |  |  | yes |
| `?? datasets/` | yes | yes | yes |
| `?? mbyolo_test.py` | yes | yes | yes |
| `?? mbyolo_test_tta.py` | yes | yes | yes |
| `?? mbyolo_train_PFIM.py` | yes |  |  |
| `?? mbyolo_train_air_ex2.py` |  | yes |  |
| `?? mbyolo_train_sarair.py` | yes | yes | yes |
| `?? selective_scan/setup.py.bak` | yes |  |  |
| `?? tools/` | yes | yes | yes |
| `?? ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-S-DS-SSM_Backbone-SSDD-ex7.yaml` | yes | yes | yes |
| `?? ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DS-SSM_Backbone-SAR_AIRcraft-ex7.yaml` | yes | yes | yes |
| `?? ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DS-SSM_Backbone-SSDD-ex7.yaml` | yes | yes | yes |
| `?? ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DS-SSM_Backbone-SSDD-ex8.yaml` | yes | yes | yes |
| `?? ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DS-SSM_Backbone-SSDD.yaml` | yes | yes | yes |
| `?? ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DS-SSM_head-SSDD.yaml` | yes | yes | yes |
| `?? ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-SSDD.yaml` | yes | yes | yes |
| `?? ultralytics/engine/trainer.py.bak.ddp_findunused.20260602_153526` | yes |  |  |
| `?? ultralytics/engine/trainer.py.bak.dsssm_freeze.20260602_152936` | yes |  |  |
| `?? ultralytics/engine/trainer.py.bak.dsssm_precise.20260602_153212` | yes |  |  |
| `?? ultralytics/engine/trainer.py.bak.pfim_fullwarmup.20260603_104323` | yes |  |  |
| `?? ultralytics/nn/modules/mamba_yolo.py.bak.20260602_152313` | yes |  |  |
| `?? ultralytics/nn/modules/mamba_yolo.py.bak.gradshape.20260602_152549` | yes |  |  |
| `?? ultralytics/nn/modules/mamba_yolo.py.bak.pfim_cuda.20260603_100820` |  |  | yes |
| `?? ultralytics/nn/modules/mamba_yolo.py.bak.pfim_cuda.20260603_101556` | yes |  |  |
| `?? ultralytics/nn/modules/mamba_yolo.py.bak.pfim_dtype.20260603_101319` |  |  | yes |
| `?? ultralytics/nn/modules/mamba_yolo.py.bak.pfim_dtype.20260603_104725` | yes |  |  |
| `?? ultralytics/nn/modules/mamba_yolo_PFIM.py` | yes |  |  |
| `?? ultralytics/nn/modules/mamba_yolo_air_ex2.py` |  | yes |  |
| `?? ultralytics/nn/modules/mamba_yolo_ex1.py` | yes | yes | yes |
| `?? ultralytics/nn/modules/mamba_yolo_ex14-15.py` |  | yes | yes |
| `?? ultralytics/nn/modules/mamba_yolo_ex19.py` |  | yes | yes |
| `?? ultralytics/nn/modules/mamba_yolo_ex2.py` | yes | yes | yes |
| `?? ultralytics/nn/modules/mamba_yolo_ex20.py` |  | yes | yes |
| `?? ultralytics/nn/modules/mamba_yolo_ex21-22.py` | yes | yes | yes |
| `?? ultralytics/nn/modules/mamba_yolo_ex23.py` | yes | yes | yes |
| `?? ultralytics/nn/modules/mamba_yolo_ex24-26.py` | yes |  | yes |
| `?? ultralytics/nn/modules/mamba_yolo_ex27-E-new.py` |  |  | yes |
| `?? ultralytics/nn/modules/mamba_yolo_ex3.py` | yes | yes | yes |
| `?? ultralytics/nn/modules/mamba_yolo_ex4.py` | yes | yes | yes |
| `?? ultralytics/nn/modules/mamba_yolo_ex5.py` | yes | yes | yes |
| `?? ultralytics/nn/modules/mamba_yolo_ex6.py` | yes | yes | yes |
| `?? ultralytics/nn/modules/mamba_yolo_ex7.py` | yes | yes | yes |
| `?? ultralytics/nn/modules/mamba_yolo_ex8.py` | yes | yes | yes |
| `?? ultralytics/nn/modules/mamba_yolo_ex9-13.py` |  | yes | yes |
| `?? ultralytics/utils/metrics.py.bak.numpy_trapz.20260603_105017` |  |  | yes |
| `?? yolov8s.pt` | yes |  |  |
| `M .gitignore` | yes | yes | yes |
