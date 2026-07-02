# Code Conflict Report

Same relative path exists with different content across at least two recovered server snapshots. These files were not merged or overwritten.

| File | 1080Ti | 3090-A | 3090-B | Bytes 1080Ti | Bytes 3090-A | Bytes 3090-B | Impact |
|---|---|---|---|---:|---:|---:|---|
| `mbyolo_train.py` | `43a95c335b55` | `3427e9cdd3b9` | `3427e9cdd3b9` | `8367` | `14714` | `14714` | `train, result` |
| `selective_scan/setup.py` | `36019920897a` | `cb67db4a7828` | `cb67db4a7828` | `6101` | `6101` | `6101` | `train, result` |
| `ultralytics.egg-info/SOURCES.txt` | `a2e99fc00f46` | `f06f69b406a7` | `ae9b3ec3ae0e` | `9328` | `8639` | `9543` | `` |
| `ultralytics/engine/trainer.py` | `33b19dade2e4` | `024587c50417` | `024587c50417` | `38418` | `35264` | `35264` | `train, result` |
| `ultralytics/nn/modules/mamba_yolo_ex23.py` | `bedbdf70e121` | `391bd24be3c6` | `391bd24be3c6` | `30539` | `47297` | `47297` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo_ex24-26.py` | `851cf25916d8` | `<missing>` | `d2c14e7773e6` | `35138` | `` | `35017` | `model, result` |
| `ultralytics/nn/tasks.py` | `e1c96ad68a32` | `c22332f2942c` | `c22332f2942c` | `44618` | `44578` | `44578` | `model, infer/val, result` |
| `ultralytics/utils/metrics.py` | `015cd5b295d7` | `015cd5b295d7` | `89480260a543` | `53518` | `53518` | `53630` | `infer/val, result` |

## Presence Differences

Same path is missing from at least one server or differs by content. This section is broader than direct conflicts and helps preserve provenance.

| File | 1080Ti | 3090-A | 3090-B | Impact |
|---|---|---|---|---|
| `mbyolo_train.py` | `43a95c335b55` | `3427e9cdd3b9` | `3427e9cdd3b9` | `train, result` |
| `mbyolo_train_PFIM.py` | `22cfb2a3b319` | `<missing>` | `<missing>` | `train, result` |
| `mbyolo_train_air_ex2.py` | `<missing>` | `9033b6598b87` | `<missing>` | `train, result` |
| `selective_scan/setup.py` | `36019920897a` | `cb67db4a7828` | `cb67db4a7828` | `train, result` |
| `selective_scan/setup.py.bak` | `36019920897a` | `<missing>` | `<missing>` | `train, result` |
| `ultralytics.egg-info/SOURCES.txt` | `a2e99fc00f46` | `f06f69b406a7` | `ae9b3ec3ae0e` | `` |
| `ultralytics/engine/trainer.py` | `33b19dade2e4` | `024587c50417` | `024587c50417` | `train, result` |
| `ultralytics/engine/trainer.py.bak.ddp_findunused.20260602_153526` | `129f4b64b42f` | `<missing>` | `<missing>` | `train, result` |
| `ultralytics/engine/trainer.py.bak.dsssm_freeze.20260602_152936` | `024587c50417` | `<missing>` | `<missing>` | `train, result` |
| `ultralytics/engine/trainer.py.bak.dsssm_precise.20260602_153212` | `41af2bbf3cc2` | `<missing>` | `<missing>` | `train, result` |
| `ultralytics/engine/trainer.py.bak.pfim_fullwarmup.20260603_104323` | `25a01ff9d6aa` | `<missing>` | `<missing>` | `train, result` |
| `ultralytics/nn/modules/mamba_yolo.py.bak.20260602_152313` | `48d1e71da219` | `<missing>` | `<missing>` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo.py.bak.gradshape.20260602_152549` | `7ab750c9836c` | `<missing>` | `<missing>` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo.py.bak.pfim_cuda.20260603_100820` | `<missing>` | `<missing>` | `3f946c62f8f2` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo.py.bak.pfim_cuda.20260603_101556` | `3f946c62f8f2` | `<missing>` | `<missing>` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo.py.bak.pfim_dtype.20260603_101319` | `<missing>` | `<missing>` | `eb4f1b402e9e` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo.py.bak.pfim_dtype.20260603_104725` | `eb4f1b402e9e` | `<missing>` | `<missing>` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo_PFIM.py` | `eb4f1b402e9e` | `<missing>` | `<missing>` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo_air_ex2.py` | `<missing>` | `48d1e71da219` | `<missing>` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo_ex14-15.py` | `<missing>` | `ba416489d819` | `ba416489d819` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo_ex19.py` | `<missing>` | `abbdc68feaec` | `abbdc68feaec` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo_ex20.py` | `<missing>` | `408ffcf127eb` | `408ffcf127eb` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo_ex23.py` | `bedbdf70e121` | `391bd24be3c6` | `391bd24be3c6` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo_ex24-26.py` | `851cf25916d8` | `<missing>` | `d2c14e7773e6` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo_ex27-E-new.py` | `<missing>` | `<missing>` | `da74bef59f22` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo_ex9-13.py` | `<missing>` | `48d1e71da219` | `48d1e71da219` | `model, result` |
| `ultralytics/nn/tasks.py` | `e1c96ad68a32` | `c22332f2942c` | `c22332f2942c` | `model, infer/val, result` |
| `ultralytics/utils/metrics.py` | `015cd5b295d7` | `015cd5b295d7` | `89480260a543` | `infer/val, result` |
| `ultralytics/utils/metrics.py.bak.numpy_trapz.20260603_105017` | `<missing>` | `<missing>` | `015cd5b295d7` | `infer/val, result` |
