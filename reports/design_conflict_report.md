# Design Conflict Report

## 1. PDF vs Current MD

Cannot determine. The requested PDFs and MD summaries are not present under workspace `docs/`, so content-level comparison is `UNKNOWN`.

Rule applied: literature PDFs must remain theory references unless a current design summary or code/experiment evidence promotes them.

## 2. MD vs Current Code

Cannot determine for the requested MD documents because they are unavailable in workspace.

Code evidence does confirm the current active module contains:

- `PFIM`
- `DeSpeckleSSM`
- `DS_VSSBlock`
- PFIM modulation of `gamma`, `write_gate`, and `delta`
- `D * x_tilde` outside the CUDA scan kernel

## 3. Code vs Experiment Conclusions

No direct conflict for IG-DS-SSM/PFIM as the current strongest line:

- Recomputed rank 1 is PFIM ex27, mAP50-95 `0.74022`.
- ex24 also ranks high, mAP50-95 `0.73571`.
- earlier non-PFIM DS-SSM ex7 ranks near top, mAP50-95 `0.7352`.

Potential concern:

- latest DDP 80-batch codex3 run reached mAP50-95 `0.73028`, lower than ex27. This suggests DDP/batch/warmup changes are not yet the best performance recipe.

## 4. PFIM vs local scale prior / GDN-style

Current status: possible extension, not current mainline.

PFIM already uses local std inside its importance estimator, but local-scale-prior/GDN-style gain control is a separate candidate design unless the missing candidate document and future experiments prove otherwise.

No direct conflict is proven, but automatic merging would create an attribution conflict: it would blur which performance gains come from PFIM versus compression-inspired scale/gain mechanisms.

## 5. Normal vs DDP Implementation Differences

Confirmed difference:

- `work_train_ddp/ultralytics/engine/trainer.py` has DDP-specific `find_unused_parameters=True`.
- `work_train_ddp/ultralytics/engine/trainer.py` has `_dsssm_freeze_warmup`.
- Hash checks show `mamba_yolo.py` and `trainer.py` differ between `work_train_single` and `work_train_ddp`.

Recommendation:

- Keep trainer differences.
- Align model module differences intentionally before future comparable experiments.

## 6. Historical Residual Code

Historical residue likely includes many `mamba_yolo_ex*.py` variants and backup PFIM modules. These are valuable archives but should not all be considered active mainline.

Examples found by code search:

- `mamba_yolo_ex1.py` through later ex-series modules.
- `mamba_yolo_PFIM.py`.
- `mamba_yolo_ex24-26.py`.
- `mamba_yolo_ex27-E-new.py` in recovered lineages.

## 7. Candidate But Unverified Code

- 20-epoch DDP warmup: implemented and runnable, but latest DDP 80-batch result is not top-ranked.
- local scale prior / GDN-style gain control: no confirmed active implementation or result.
- SAR-specific loss / augmentation: `UNKNOWN` in this audit.

## 8. Designs Not To Enter Current Mainline Yet

- local scale prior
- GDN-style gain normalization
- SAR-specific loss/augmentation, unless verified in code and experiment
- any early PDF idea not reflected in current code/results
- any ex-series historical variant without a clear result row and config mapping
