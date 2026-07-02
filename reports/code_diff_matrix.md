# Code Diff Matrix

## Relative to Official Shared HEAD

| File | 1080Ti | 3090-A | 3090-B | Impact flags |
|---|---|---|---|---|
| `.gitignore` | `M +1/-0 (1 lines)` | `M +1/-0 (1 lines)` | `M +1/-0 (1 lines)` | `` |
| `mbyolo_test.py` | `A(untracked) 234 lines, 10865 bytes` | `A(untracked) 234 lines, 10865 bytes` | `A(untracked) 234 lines, 10865 bytes` | `` |
| `mbyolo_test_tta.py` | `A(untracked) 96 lines, 4261 bytes` | `A(untracked) 96 lines, 4261 bytes` | `A(untracked) 96 lines, 4261 bytes` | `` |
| `mbyolo_train.py` | `M +139/-17 (156 lines)` | `M +324/-18 (342 lines)` | `M +324/-18 (342 lines)` | `train, result` |
| `mbyolo_train_PFIM.py` | `A(untracked) 353 lines, 14684 bytes` | `` | `` | `train, result` |
| `mbyolo_train_air_ex2.py` | `` | `A(untracked) 253 lines, 11883 bytes` | `` | `train, result` |
| `mbyolo_train_sarair.py` | `A(untracked) 165 lines, 5527 bytes` | `A(untracked) 165 lines, 5527 bytes` | `A(untracked) 165 lines, 5527 bytes` | `train, result` |
| `selective_scan/setup.py` | `M +1/-1 (2 lines)` | `` | `` | `train, result` |
| `tools/convert_sarair_coco2yolo.py` | `A(untracked) 112 lines, 3474 bytes` | `A(untracked) 112 lines, 3474 bytes` | `A(untracked) 112 lines, 3474 bytes` | `train, result` |
| `tools/prepare_ssdd_bbox_voc_to_yolo.py` | `A(untracked) 157 lines, 4098 bytes` | `A(untracked) 157 lines, 4098 bytes` | `A(untracked) 157 lines, 4098 bytes` | `train, result` |
| `tools/prepare_ssdd_coco_to_yolo.py` | `A(untracked) 200 lines, 5513 bytes` | `A(untracked) 200 lines, 5513 bytes` | `A(untracked) 200 lines, 5513 bytes` | `train, result` |
| `tools/prepare_ssdd_yolo.py` | `A(untracked) 202 lines, 5507 bytes` | `A(untracked) 202 lines, 5507 bytes` | `A(untracked) 202 lines, 5507 bytes` | `train, result` |
| `ultralytics/cfg/default.yaml` | `M +3/-0 (3 lines)` | `M +3/-0 (3 lines)` | `M +3/-0 (3 lines)` | `train, result` |
| `ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-S-DS-SSM_Backbone-SSDD-ex7.yaml` | `A(untracked) 50 lines, 1648 bytes` | `A(untracked) 50 lines, 1648 bytes` | `A(untracked) 50 lines, 1648 bytes` | `model, train, result` |
| `ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DS-SSM_Backbone-SAR_AIRcraft-ex7.yaml` | `A(untracked) 50 lines, 1648 bytes` | `A(untracked) 50 lines, 1648 bytes` | `A(untracked) 50 lines, 1648 bytes` | `model, train, result` |
| `ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DS-SSM_Backbone-SSDD-ex7.yaml` | `A(untracked) 50 lines, 1648 bytes` | `A(untracked) 50 lines, 1648 bytes` | `A(untracked) 50 lines, 1648 bytes` | `model, train, result` |
| `ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DS-SSM_Backbone-SSDD-ex8.yaml` | `A(untracked) 50 lines, 1656 bytes` | `A(untracked) 50 lines, 1656 bytes` | `A(untracked) 50 lines, 1656 bytes` | `model, train, result` |
| `ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DS-SSM_Backbone-SSDD.yaml` | `A(untracked) 50 lines, 1651 bytes` | `A(untracked) 50 lines, 1651 bytes` | `A(untracked) 50 lines, 1651 bytes` | `model, train, result` |
| `ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DS-SSM_head-SSDD.yaml` | `A(untracked) 39 lines, 1873 bytes` | `A(untracked) 39 lines, 1873 bytes` | `A(untracked) 39 lines, 1873 bytes` | `model, train, result` |
| `ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-SSDD.yaml` | `A(untracked) 38 lines, 1244 bytes` | `A(untracked) 38 lines, 1244 bytes` | `A(untracked) 38 lines, 1244 bytes` | `model, train, result` |
| `ultralytics/engine/trainer.py` | `M +101/-1 (102 lines)` | `` | `` | `train, result` |
| `ultralytics/nn/modules/__init__.py` | `M +2/-1 (3 lines)` | `M +2/-1 (3 lines)` | `M +2/-1 (3 lines)` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo.py` | `M +582/-1 (583 lines)` | `M +582/-1 (583 lines)` | `M +582/-1 (583 lines)` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo_PFIM.py` | `A(untracked) 999 lines, 34723 bytes` | `` | `` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo_air_ex2.py` | `` | `A(untracked) 796 lines, 29010 bytes` | `` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo_ex1.py` | `A(untracked) 950 lines, 32326 bytes` | `A(untracked) 950 lines, 32326 bytes` | `A(untracked) 950 lines, 32326 bytes` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo_ex14-15.py` | `` | `A(untracked) 778 lines, 30057 bytes` | `A(untracked) 778 lines, 30057 bytes` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo_ex19.py` | `` | `A(untracked) 798 lines, 30621 bytes` | `A(untracked) 798 lines, 30621 bytes` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo_ex2.py` | `A(untracked) 1029 lines, 35861 bytes` | `A(untracked) 1029 lines, 35861 bytes` | `A(untracked) 1029 lines, 35861 bytes` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo_ex20.py` | `` | `A(untracked) 854 lines, 33134 bytes` | `A(untracked) 854 lines, 33134 bytes` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo_ex21-22.py` | `A(untracked) 1001 lines, 39422 bytes` | `A(untracked) 1001 lines, 39422 bytes` | `A(untracked) 1001 lines, 39422 bytes` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo_ex23.py` | `A(untracked) 843 lines, 30539 bytes` | `A(untracked) 1178 lines, 47297 bytes` | `A(untracked) 1178 lines, 47297 bytes` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo_ex24-26.py` | `A(untracked) 1007 lines, 35138 bytes` | `` | `A(untracked) 1005 lines, 35017 bytes` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo_ex27-E-new.py` | `` | `` | `A(untracked) 1004 lines, 34265 bytes` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo_ex3.py` | `A(untracked) 1042 lines, 36414 bytes` | `A(untracked) 1042 lines, 36414 bytes` | `A(untracked) 1042 lines, 36414 bytes` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo_ex4.py` | `A(untracked) 1042 lines, 36433 bytes` | `A(untracked) 1042 lines, 36433 bytes` | `A(untracked) 1042 lines, 36433 bytes` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo_ex5.py` | `A(untracked) 1063 lines, 37140 bytes` | `A(untracked) 1063 lines, 37140 bytes` | `A(untracked) 1063 lines, 37140 bytes` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo_ex6.py` | `A(untracked) 680 lines, 24416 bytes` | `A(untracked) 680 lines, 24416 bytes` | `A(untracked) 680 lines, 24416 bytes` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo_ex7.py` | `A(untracked) 680 lines, 24416 bytes` | `A(untracked) 680 lines, 24416 bytes` | `A(untracked) 680 lines, 24416 bytes` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo_ex8.py` | `A(untracked) 796 lines, 29010 bytes` | `A(untracked) 796 lines, 29010 bytes` | `A(untracked) 796 lines, 29010 bytes` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo_ex9-13.py` | `` | `A(untracked) 796 lines, 29010 bytes` | `A(untracked) 796 lines, 29010 bytes` | `model, result` |
| `ultralytics/nn/tasks.py` | `M +8/-4 (12 lines)` | `M +6/-2 (8 lines)` | `M +6/-2 (8 lines)` | `model, infer, result` |
| `ultralytics/utils/metrics.py` | `` | `` | `M +3/-1 (4 lines)` | `infer, result` |

## Cross-Server Code Hash Differences

This section includes all collected code/config/doc paths outside experiment and dataset directories, including untracked files.

| File | 1080Ti | 3090-A | 3090-B |
|---|---|---|---|
| `mbyolo_train.py` | `43a95c335b55` | `3427e9cdd3b9` | `3427e9cdd3b9` |
| `mbyolo_train_PFIM.py` | `22cfb2a3b319` | `<missing>` | `<missing>` |
| `mbyolo_train_air_ex2.py` | `<missing>` | `9033b6598b87` | `<missing>` |
| `selective_scan/setup.py` | `36019920897a` | `cb67db4a7828` | `cb67db4a7828` |
| `ultralytics/engine/trainer.py` | `33b19dade2e4` | `024587c50417` | `024587c50417` |
| `ultralytics/nn/modules/mamba_yolo_PFIM.py` | `eb4f1b402e9e` | `<missing>` | `<missing>` |
| `ultralytics/nn/modules/mamba_yolo_air_ex2.py` | `<missing>` | `48d1e71da219` | `<missing>` |
| `ultralytics/nn/modules/mamba_yolo_ex14-15.py` | `<missing>` | `ba416489d819` | `ba416489d819` |
| `ultralytics/nn/modules/mamba_yolo_ex19.py` | `<missing>` | `abbdc68feaec` | `abbdc68feaec` |
| `ultralytics/nn/modules/mamba_yolo_ex20.py` | `<missing>` | `408ffcf127eb` | `408ffcf127eb` |
| `ultralytics/nn/modules/mamba_yolo_ex23.py` | `bedbdf70e121` | `391bd24be3c6` | `391bd24be3c6` |
| `ultralytics/nn/modules/mamba_yolo_ex24-26.py` | `851cf25916d8` | `<missing>` | `d2c14e7773e6` |
| `ultralytics/nn/modules/mamba_yolo_ex27-E-new.py` | `<missing>` | `<missing>` | `da74bef59f22` |
| `ultralytics/nn/modules/mamba_yolo_ex9-13.py` | `<missing>` | `48d1e71da219` | `48d1e71da219` |
| `ultralytics/nn/tasks.py` | `e1c96ad68a32` | `c22332f2942c` | `c22332f2942c` |
| `ultralytics/utils/metrics.py` | `015cd5b295d7` | `015cd5b295d7` | `89480260a543` |
