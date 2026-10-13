# Unified detection adapter — 2026-10-12

This is an independent protocol implementation, not a continuation or reuse of
the historical 150/200-epoch or new 300-epoch result tracks. No original source,
dataset, weights, experiment, or queue is edited by these scripts. Server model
execution remains the coordinator's responsibility; local AST checks do not
constitute a successful CUDA probe or trained result.

## CLI

```bash
CUDA_VISIBLE_DEVICES=GPU-<full-assigned-uuid> <existing-env>/bin/python detection/run_detection.py \
  --source-root <frozen-project-ultralytics-root> \
  --data <audited-data.yaml> --dataset-manifest <data_content.json-or-descriptor.json> \
  --dataset-id ssdd --method Original --seed 0 \
  --output <new-absolute-run-directory> --gpu-uuid GPU-<full-assigned-uuid> --mode probe
```

Change only `--mode train` and the fresh output directory for formal training.
No automatic resume, checkpoint transfer, batch-size fallback, or CPU fallback
exists. The coordinator must own the existing campaign GPU lock and its new
queue lock before launching; the runner does not acquire a competing lock. It
checks the exact GPU UUID and rejects any foreign compute PID. It never kills
another process. `--source-manifest` optionally requires predeclared source file
hashes; every run also records and rechecks actual source and runner hashes.

Supported methods are `Original`, `Matched075`, `Full`, `YOLOv8n`, and `YOLOv8s`
(their legacy identifiers are aliases). YOLOv8s is available for its existing
candidate placements, especially SADD; it is **not an extra SSDD candidate**.
Only seeds 0 and 1 are accepted. A seed1 launch still requires the coordinator's
new campaign policy; this adapter does not issue or bypass any old seed barrier.

## Frozen numerical policy

- Physical batch16, accumulation4, nominal64. The last group of every epoch is
  flushed using its actual samples; no data are dropped. Gradients from the
  native sample-scaled detection loss are divided once by that actual count.
  Physical-microbatch BatchNorm is not equivalent to physical batch64.
- AdamW `lr=2e-4`, betas `(0.937,0.999)`, eps `1e-8`; ordinary weights decay
  `5e-4`, bias-name aliases and norm parameters decay zero. Global gradient
  norm clip10, no optimizer auto-selection or implicit LR/WD scaling.
- Up to150 epochs, validation patience30; exactly3 complete actual epochs of
  linear LR warmup, with no100-step minimum and no special bias-LR peak. Then
  linear decay reaches `2e-6` at the planned150-epoch endpoint. Early stopping
  does not rescale the remaining schedule. Module warmup is zero.
- Scratch FP32 throughout; AMP/TF32 off. Mamba common tensors outside P3 are
  same-seed paired to Original, and stride-probe BN statistics are reset.
  All variants use identical dataset-specific splits and the augmentation
  dictionary in `detector_spec.py`; mosaic closes at epoch136. The ordinary
  PyTorch loader uses seed-bound shuffling, no dropped tail, and no persistent
  workers, so the mosaic change reaches worker copies.
- Every epoch selects on EMA validation `0.1*AP50+0.9*AP50_95`, keeping the
  earliest exact tie. Only after training reaches an allowed terminal condition
  is the selected FP32 checkpoint reloaded and evaluated on the independent
  test split. Validation/test both use conf0.001, NMS IoU0.7, max_det300, batch16
  and the same unfused FP32 metric path. P/R summarize the maximum-F1 curve;
  they are not a prespecified deployment threshold.

## Dataset binding

`--data` accepts an explicit local YAML with flat `images/train`, `images/val`,
`images/test` and corresponding `labels/<split>` directories. Download commands,
implicit dataset lookup, and mutable model templates are not supported.

`--dataset-manifest` supports:

- The old explicit list of `{split,name,image_sha256,label_sha256}` records,
  together with `--data`.
- A descriptor with `yaml`/`data_yaml`, optional YAML hash, `root`, `names`,
  `counts`, and either top-level `records` or a hash-bound `manifest` path.
- The new SADD `rgsp-unified-data-v1` descriptor from this campaign's data
  preparation. Its exact protocol ID is
  `sadd-full2966-posneg-stratified-80-10-10-rgbgroup-v1`.

SSDD is730/186/232; new full positive+negative SADD is2372/296/298; Ship is
31735/3970/3974. AIR is held with its unresolved source-label conflict. SADD
positive-only795/44/44 and SSDD928/232 TEST-as-VAL cannot pass this adapter.
Image/class/box checks are fail-closed; images/labels are never repaired.
Source inputs are hashed, and only symlinks/cache files are created in the new
run directory. Probe materializes/verifies only `train`; it does not enumerate
or open val/test image or label files. Reading their manifest metadata is not
model evaluation. Formal preflight checks all three splits for data integrity,
but test has no loader or model evaluation until final best-checkpoint testing.

## Probe and result contract

Probe defaults to2 timing-warmup optimizer updates plus4 measured updates. These
timing exclusions are separate from the3-epoch LR warmup. It emits
`result.json: status=PROBE_PASS`, `test_evaluated=false`, `val_evaluated=false`,
`method`, `protocol_sha256`, `estimated_full_seconds`, actual sample throughput,
peak GPU memory, and controls. Estimated time covers planned150-epoch training
updates only; validation, checkpoints, data I/O and early stopping are excluded.
Probe metrics are not paper performance results.

Formal `result.json` has `status=TRAINING_COMPLETE`, `metrics` with
`P/R/AP50/AP75/AP50_95` in0–1 units, percent aliases, the protocol/source/runner/
data hashes, and `checkpoint={path,sha256}`. `completion.json` independently binds
the result SHA. `weights/best.pt` and `last.pt` retain FP32 model/EMA/optimizer
state; no framework FP16 optimizer stripping is called. They are trusted local
task checkpoints, not safe formats for loading an arbitrary downloaded pickle.
The first best reload verifies tensor hash, floating dtypes, and RGSP controls.

`updates.jsonl` and `optimizer_ledger.jsonl` record actual groups, samples,
optimizer updates, LR, gradients, validation values, controls and stop reason.
OOM is `BLOCKED_OOM`; no lower-batch retry is allowed. Failures produce no valid
completion receipt. Partial/failed outputs are preserved and cannot be silently
reused as fresh runs.

## Complexity and other candidates

`covered_flops.py` provides
`profile_model(model, imgsz=640, inputs=None)`. The default is dense unfused
batch1 FP32 detection; an explicit input tuple also supports full classification
or two-input change-detection models. It profiles independent model/input copies
and restores hooks, buffers, modes, RNG and monkeypatches. THOP-derived counts
plus actual scan/einsum/CrossMerge contributions are **covered-operation
estimates**, not exhaustive FLOPs, hardware instructions or a speed benchmark.
The returned coverage/unsupported fields must accompany every displayed value.
Missing/failed coverage is null with its reason and does not fabricate zero or
invalidate an otherwise completed performance result. The runner verifies its
initial training state was not modified by the profile.

`CANDIDATE_AUDIT.md` covers all67 existing detection candidate placements and42
distinct external methods. Source availability, historical profiler environment,
exact scale, and a unified training adapter are different readiness levels.
The five-method adapter does not make all42 external methods runnable. AIR
historical candidates and proposals outside the current CSV remain separate.

## Verification boundary

Initial implementation: all four Python files were parsed with Python's AST
parser, with no imports or execution of the training modules. A successful
server `probe` and subsequent end-to-end train/checkpoint/test evidence remain
required. No SSH, CUDA model execution, training, or queue launch was performed
by the detection-adapter author. The formal shared project memory is not copied
or updated here; the coordinator owns the final evidence-backed memory delta.
