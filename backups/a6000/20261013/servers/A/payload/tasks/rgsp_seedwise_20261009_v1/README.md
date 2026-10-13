# RGSP finite, globally seed-ordered execution

This independent directory does not change old campaign outputs or the original
trainer. Execution is server-only. The root operator must audit each server and
freeze the whole plan before launching it. These files do not themselves select
servers, run SSH, install software, interrupt any process, or create a service.

## Entry points

- `manifest.py --plan /absolute/plan.json --write-configs` writes exact YAMLs once.
- `run_lane.py --plan /absolute/plan.json --lane a --verify-only` verifies the
  plan and source hashes without touching a GPU.
- `run_lane.py --plan /absolute/plan.json --lane a` executes that finite list.
- `launch_lane.sh /absolute/python /absolute/plan.json a` is the same foreground
  entrypoint, ready for the operator to launch through `nohup` after inspection.
- `manifest.py --plan /absolute/plan.json --issue-barrier --receipts-dir
  /absolute/collected_seed0_receipts` is a **trusted coordinator** action, never
  called by a lane. This is a hash-bound barrier, not a cryptographic signature.
- `train_trial.py --plan /absolute/plan.json --trial ID --gpu-uuid GPU-...`
  runs one detection trial. `CUDA_VISIBLE_DEVICES` must already equal that UUID.
  The lane normally calls this itself.
- `campaign_dispatch.py --plan /absolute/plan.json --host-key a --detach`
  starts a finite server dispatcher: explicit dependencies, the assigned seed0
  lane, a verified global seed0 barrier, then the assigned seed1 lane. Without
  `--detach` it stays foreground; `--verify-only` does not launch any work.
- `verify_receipts.py --plan /absolute/plan.json --host-key a --seed 0`
  emits complete read-only JSON for the entire assigned subset, verifying the
  exact ordered semantic artifacts and every artifact hash on its producing
  host. `--require-complete` exits 2 for missing/invalid/not-ready results.

There is no implicit resume. An explicit `run_lane.py ... --resume-rich ID
/absolute/current_run/weights/raw_fp32_last.pt` restores one compatible raw-FP32
checkpoint. The direct trainer accepts `--resume-rich /absolute/path`. Legacy
EMA/FP16 `last.pt` is refused. Terminal checkpoint recovery/finalization is not
automatic; failed jobs stop and preserve their partial outputs for inspection.

## Immutable plan schema

Every path is an absolute path on its assigned server. The same plan file is
distributed to all hosts; identical server-side task roots simplify this. All
IDs in `seed0_trial_ids` must equal **all** seed0 trial IDs, including unassigned
and not-ready items. Merely finishing one lane cannot unlock seed1.

```json
{
  "schema": "rgsp-seedwise-v1",
  "campaign": "rgsp_seedwise_20261009_v1",
  "campaign_root": "/explicit/new/task/root",
  "source_root": "/explicit/new/task/root/code",
  "python": "/explicit/already-installed/python",
  "barrier_path": "/explicit/new/task/root/seed0_barrier.json",
  "receipts_dir": "/explicit/new/task/root/receipts",
  "code_hashes": [{"path": "/explicit/new/task/root/code/example.py", "sha256": "64 lowercase hex characters"}],
  "datasets": {
    "ssdd_seed0": {
      "protocol": "ssdd",
      "root": "/explicit/new/task/root/data/ssdd_seed0",
      "yaml": "/explicit/new/task/root/data/ssdd_seed0.yaml",
      "yaml_sha256": "64 lowercase hex characters",
      "manifest": "/explicit/new/task/root/data/ssdd_seed0_manifest.json",
      "manifest_sha256": "64 lowercase hex characters",
      "counts": {"train": 730, "val": 186, "test": 232}
    }
  },
  "seed0_trial_ids": ["ssdd_full_seed0"],
  "trials": [{
    "id": "ssdd_full_seed0", "seed": 0, "kind": "detection",
    "dataset": "ssdd_seed0",
    "spec": {"control": "full", "placement": "p3", "state_dim": 16,
             "trend_kernel_size": 3, "stats_kernel_size": 3},
    "config": "/explicit/new/task/root/configs/ssdd_full_seed0.yaml",
    "run_dir": "/explicit/new/task/root/runs/ssdd_full_seed0"
  }],
  "lanes": [{
    "id": "a", "hostname": "verified-hostname", "gpu_uuid": "GPU-verified-uuid",
    "output_dir": "/explicit/new/task/root/lanes/a",
    "idle_seconds": 120, "idle_timeout_seconds": 3600,
    "max_idle_memory_mib": 1024, "max_idle_utilization_percent": 5,
    "trial_ids": ["ssdd_full_seed0"]
  }]
}
```

The example is schema documentation, not a runnable server assignment. Real
code hashes must include all imported project source, YAML defaults, original
training/evaluation protocol source, `tools/validate_rgsp_expanded.py`, and these
runner files. No file is guessed from old chats or credentials.

Dataset manifests have the existing list format:
`[{"split":"train","name":"image.jpg","image_sha256":"...","label_sha256":"..."}]`.
Images and labels live under `root/images/{train,val,test}` and
`root/labels/{train,val,test}`. Every listed image/label hash, exact image set,
split count, YAML destination, and cross-split byte identity is checked.
This is not a scene-disjoint or near-duplicate audit. Each dataset record may
select `protocol: "ssdd"` or `"sadd"`, independent of its dictionary key.

Detection configuration supports P2, P3, P4, d8/d16/d32, and independent trend
and statistics windows 3/5/7. P2/P3 replace one actual block. P4's nominal
6 ordinary + 3 RGSP repetitions at depth .33 become **2 ordinary + 1 RGSP**;
all downstream neck indices are rebuilt. Original uses `control: "original"`,
`placement: "baseline"` and no RGSP block. The runtime checks the actual count.
The `cnn_yolov8n` reference also uses `placement: "baseline"` and zero RGSP
blocks. Its explicit architecture is the frozen repository YOLOv8 n-scale
template with nc=1, not a downloaded or mutable installed-package config. It
uses the same dataset/training protocol, not a claim of parameter matching.

Additional trial kinds:

- `command`: explicit `command` argv list (absolute executable), absolute `cwd`,
  unique `run_dir`, and explicit `success_artifacts` paths inside that run.
  A nonzero exit, existing partial run, or missing/empty artifact stops the lane.
  The coordinator must specify all semantic completion artifacts, not only a log.
- `reuse`: no training. `reuse` contains `train_complete`, `eval_test`,
  `evaluation_complete`, `best`, `protocol`, and `audit`, each as `{path,sha256}`.
  They are all hashed. The audit JSON must say `status: "verified"`, name this
  `trial_id`, explicitly affirm `protocol_compatible: true`, and retain a
  nonempty `limitations` description. Train/test/completion must agree on the
  exact best hash and seed. Old interrupted-training caveats are not erased.
- `not_ready`: records an intended but not implemented experiment. It always
  blocks execution and the global barrier. It cannot silently disappear from
  the matrix. Changing an immutable plan requires a separately reviewed version;
  old receipts do not automatically complete a changed plan.

Only seeds 0 and 1 are accepted. A separate smoke plan sets `smoke_only: true`,
uses independent output directories, and runs 2 epochs; its different plan hash
cannot unlock a formal campaign. Never put smoke outputs into formal runs.

### Deferred external detection data

External trials use `kind: "command"`, `runner_kind: "deferred_detection"`, and
the same `dataset`, `spec`, `config`, and `run_dir` fields as detection trials.
Their explicit command invokes `train_trial.py --plan ... --trial ...
--gpu-uuid ...`; `--write-configs` includes these trials. `success_artifacts`
must contain exactly these absolute paths beneath that run, in this order:
`provenance.json`, `train_complete.json`, `eval_test.json`,
`evaluation_complete.json`, `weights/best.pt`, `weights/raw_fp32_last.pt`,
`controls_by_epoch.jsonl`, `trial_complete.json`. `manifest.detection_artifacts`
returns the same list, so the trainer and command lane cannot disagree.

```json
{
  "deferred": true,
  "descriptor": "/explicit/task/data/sar_shipdataset/dataset.json",
  "audit": "/explicit/task/data/sar_shipdataset/source_audit.json",
  "expected_identity": "SAR-ShipDataset",
  "expected_nc": 1,
  "protocol": "ssdd"
}
```

The other supported identity is `SAR-AIRcraft1.0`, with `expected_nc: 7`.
`readiness` may explicitly override the fixed default `readiness.json` beside
the descriptor. Before importing CUDA libraries, the trainer waits at most
seven days, checking every 30 seconds without writing into its trial directory.
The producer publishes the descriptor and audit first, then readiness last:

- Descriptor: the standard resolved `root`, `yaml`, `yaml_sha256`, `manifest`,
  `manifest_sha256`, positive `counts` for train/val/test, and `protocol: "ssdd"`,
  plus explicit `nc` and class-ordered `names`.
- Source audit: `status: "verified"`, exact `dataset_identity`, `nc`,
  `manifest_sha256`, matching image `counts`, and class-ordered `names`.
  Additional source/scene and COCO category mapping fields are retained by its
  file hash.
- Readiness: `status: "verified_ready"`, exact `dataset_identity`, `nc`,
  `descriptor_sha256`, `manifest_sha256`, and `source_audit_sha256`.

Malformed or hash-inconsistent ready publications fail closed. Producer states
`failed`, `invalid`, and `rejected` also stop the lane. A valid publication is
followed by the complete staged image/label hash audit, a stable-publication
recheck, a fresh code audit and a fresh GPU occupation check. The three publication
file hashes enter provenance. Audit status is a trusted producer assertion, not
an independent proof of scene disjointness. Only the descriptor's hash-bound YAML
provides class order; historical aircraft YAMLs are never substituted. The model
template starts with nc=1 and the trainer overrides it from the audited dataset.
Only these deferred configs use `Detect: [nc]` via
`config_payload(spec, dynamic_nc=True)`; default SSDD/SADD/smoke configs keep
their original numeric `[1]`. Descriptor, source audit, dataset YAML, runtime
dataset/model class mappings, model metadata and actual detection-head nc are
all compared, including after loading the selected checkpoint.
Both external datasets use the same 150/30, 640, batch16, FP32 SSDD protocol; old
150/300-epoch scores do not count as completion evidence for these new arms.

### Finite dispatch and receipt collection

`dispatch_map` has explicit keys `a`, `b`, `c`, `f0`, `f1`. Each entry contains
`hostname`, `gpu_uuid`, `seed0_lane`, and `seed1_lane`; selected lanes must match
that host/GPU and contain only their indicated seed. Optional `requires` is a
list of `{path, equals: {field: exact_value}}` completion dependencies.
F entries must require their own legacy completion marker with
`status: "adopted_and_scheduler_retired"` and an explicit `finalizer_launch`
record. Missing launch evidence, a dead finalizer without completion, or a bad
completion value stops dispatch. Successful dependencies still wait for that
finalizer to exit and the campaign GPU lock to become available.

Detached dispatch retains an exclusive inherited lock and records PID/start
time/argv, logs and phase completion in `dispatch/<host-key>/attempt_.../`.
Current status is `dispatch/<host-key>/status.json`. A failed child stops all
later phases; there is no retry or signal. Dependency and global-barrier waits
are each finite at seven days. Intended not-ready seed0 rows independently forbid
seed1 even if an invalid coordinator attempts to manufacture a barrier.

The collection interface is the verifier's **complete stdout JSON**, not a
truncated terminal excerpt. For example, run on the producing host:

```sh
/absolute/python -B /absolute/runner/verify_receipts.py \
  --plan /absolute/plan.json --host-key b --seed 0
```

The authorized coordinator captures all stdout in its evidence store, checks
`plan_sha256`, `requested_trial_ids`, `all_requested_complete`, and every row,
then combines all producing-host subsets. `global_seed0_unverified_in_this_snapshot`
remains explicit; a local lane can never assert global completion. The verifier
does not write a receipt, create a barrier, use SSH, or allocate a GPU.

## Completion and barrier contract

A detection trial emits `provenance.json`, `runtime_audit.json`, per-epoch
control checks, legacy-format best/last weights, raw-FP32 checkpoints,
`train_complete.json`, best-hash-bound `eval_test.json`,
`evaluation_complete.json`, `trial_complete.json`, and then a receipt.

`receipts/<trial_id>.json` has:

```json
{
  "schema": "rgsp-seedwise-v1", "status": "complete",
  "plan_sha256": "SHA256(canonical entire plan)",
  "trial_id": "exact ID", "trial_sha256": "SHA256(canonical trial)",
  "seed": 0, "finished_utc": "...",
  "artifacts": [{"path": "/absolute/evidence", "sha256": "..."}],
  "details": {"kind": "detection", "smoke_only": false}
}
```

Canonical means UTF-8 JSON, sorted keys, separators `(',', ':')`, no ASCII
escaping, and no NaN. The trusted coordinator collects remote receipts **only
after checking their artifact hashes on the producing host**. It then calls
`manifest.py --issue-barrier`. The tool requires every intended seed0 receipt,
checks trial/plan identity and full coverage, and refuses not-ready items.
The marker is `{"payload": {...all receipts...}, "payload_sha256": "..."}`.
Copy that identical marker to each host's explicit `barrier_path`. Both lane and
detection trainer verify it before every seed1 trial. It is not an authorization
or anti-tampering security mechanism; it prevents partial-lane/old-plan unlocks.

## Protocol and restart limitations

SSDD is 150 epochs/patience30, SADD 200/patience40. Both preserve scratch,
640, batch16, FP32, AdamW2e-4, linear schedule, nbs64, 3-epoch optimizer warmup,
zero module warmup, source augmentations, val fitness selection, and independent
test conf .001/IoU .6/workers0. Original best/last serialization remains the
existing EMA/FP16 format to avoid silently changing legacy evaluation precision.
The sole visible device is selected without allowing Ultralytics to replace the
UUID mask with a physical ordinal. Setup verifies the current process on that
exact UUID and verifies the reset BN statistics. Same seed is not common-tensor
alignment between different architectures, state sizes, placements or trend
kernel shapes; per-tensor initialization hashes retain this limitation explicitly.
`cuda_uuid_compat.py` is required in the frozen runner package. It repairs only
the verified PyTorch 1.13 zero-NVML-count/one-native-CUDA-device UUID-parser case
inside the current process; the torch installation and UUID mask stay unchanged.

Additional raw-FP32 checkpoints retain raw model, EMA, optimizer, scheduler,
scaler, RNG, early stopping, dataloader generator and results CSV; first-epoch
serialization and explicit restoration hash checks fail closed. They improve
state recovery, **not** bitwise continuation: inherited worker prefetch and the
gradient-accumulation step cursor are not replayed. The original trainer is not
patched. No terminal/legacy checkpoint is silently resumed or restarted.

Each lane retains exclusive lane and campaign/GPU-UUID locks, checks actual host
and UUID, and requires no compute process plus low memory/utilization for the
configured continuous idle interval before each new trial. It never takes over
a busy device, and always stops on a failed task. Locks coordinate this campaign;
unrelated software may not honor them, so the final GPU check still has an
unavoidable cross-process launch race on a shared server.

Implementation was written and statically reviewed on Windows. CUDA parsing,
training, restore/evaluation smoke tests must be run only on audited servers;
their actual receipts, not this README, determine verification status.
