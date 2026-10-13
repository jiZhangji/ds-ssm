#!/usr/bin/env python3
"""Frozen SAMPLE source-only probe/training. There is deliberately no test mode.

Both modes require a coordinator-produced sample-run-freeze-v1 JSON containing:
protocol_sha256, manifest_sha256, audit_sha256, source_root,
source_files=[{path: relative-to-source-root, sha256: ...}], and
support_files=[{path: relative-to-this-directory, sha256: ...}]. All source .py
files and this runner, protocol, factory and its local dependencies are bound.

Formal training additionally requires sample-training-approval-v1 JSON with
approved=true, authorized_by="user", evidence=<current authorization reference>,
freeze_sha256, protocol_sha256, optimizer_updates=10000, gpu_uuid, and
runs=[{variant, seed, output: absolute-new-output-directory}]. A probe never
creates or implies training approval. This executable cannot evaluate real data.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import random
import socket
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
GPU_UUID = "GPU-12cf4841-46b9-94d1-2715-9932134cf964"
LEGACY_LOCK = Path("/home/powerleader/ig-ds-ssm/tasks/rgsp_seedwise_20261009_v1/locks") / (GPU_UUID + ".lock")
VARIANTS = ("original", "matched_scale075", "full")
BATCH, UPDATES, TRAIN_COUNT, TEST_COUNT, SIZE = 36, 10000, 806, 539, 128


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def absolute(value):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"An absolute server path is required: {value}")
    return path.resolve()


def within(root, relative):
    if Path(relative).is_absolute():
        raise ValueError("Frozen inventory paths must be relative")
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError(f"Frozen path escapes its root: {relative}")
    return path


def verify_inventory(root, rows):
    seen = set()
    for row in rows:
        path = within(root, row["path"])
        if path in seen or not path.is_file() or sha(path) != row["sha256"]:
            raise RuntimeError(f"Missing, duplicate, or changed frozen file: {path}")
        seen.add(path)
    return seen


def verify_inputs(args):
    protocol_path = HERE / "protocol.json"
    recipe = read_json(protocol_path)
    expected = {
        "schema": "sample-fixed-update-protocol-v1", "dataset_schema": "sample-domain-angle-v1",
        "classes": 10, "input_size": SIZE, "physical_batch_size": BATCH,
        "gradient_accumulation": 1, "optimizer": "SGD", "learning_rate": 0.001,
        "momentum": 0.9, "dampening": 0.0, "nesterov": False, "weight_decay": 0.0005,
        "learning_rate_schedule": "constant", "optimizer_updates": UPDATES,
        "module_warmup_updates": 0, "loss": "cross_entropy", "label_smoothing": 0.0,
        "gradient_clipping": None, "augmentation": "none", "validation": None,
        "variants": list(VARIANTS),
    }
    if any(recipe.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Protocol differs from the implemented fixed recipe")
    freeze = read_json(args.freeze_file)
    if freeze.get("schema") != "sample-run-freeze-v1":
        raise RuntimeError("Explicit sample-run-freeze-v1 is required")
    if absolute(freeze["source_root"]) != args.source_root:
        raise RuntimeError("Frozen source root differs")
    for path, key in ((protocol_path, "protocol_sha256"), (args.manifest, "manifest_sha256"),
                      (args.manifest.parent / "audit.json", "audit_sha256")):
        if sha(path) != freeze.get(key):
            raise RuntimeError(f"Changed protocol or data artifact: {path}")
    source_files = verify_inventory(args.source_root, freeze["source_files"])
    all_python = {path.resolve() for path in args.source_root.rglob("*.py")}
    if not all_python or not all_python.issubset(source_files):
        raise RuntimeError("Frozen source inventory must cover every Python source file")
    if any(Path(row["path"]).suffix != ".py" and row["path"] != "protocol.json"
           for row in freeze["support_files"]):
        raise RuntimeError("Support inventory accepts code/protocol only; never add sealed data manifests")
    support_files = verify_inventory(HERE, freeze["support_files"])
    required_support = {HERE / name for name in ("train_sample.py", "protocol.json", "model_factory.py",
                                                 "manifest.py", "cross_data.py", "cuda_uuid_compat.py")}
    if not required_support.issubset(support_files):
        raise RuntimeError("Frozen support inventory omits runner/factory/protocol/import dependencies")
    required_source = args.source_root / "tools/run_shared_sar_tasks_v2.py"
    if required_source not in source_files:
        raise RuntimeError("Runtime source is absent from the freeze")
    approval = None
    if args.mode == "train":
        if args.approval_file is None:
            raise RuntimeError("Formal training requires explicit --approval-file; a probe is not approval")
        approval = read_json(args.approval_file)
        checks = {
            "schema": "sample-training-approval-v1", "approved": True, "authorized_by": "user",
            "freeze_sha256": sha(args.freeze_file), "protocol_sha256": sha(protocol_path),
            "optimizer_updates": UPDATES, "gpu_uuid": GPU_UUID,
        }
        if any(approval.get(key) != value for key, value in checks.items()) or not str(approval.get("evidence", "")).strip():
            raise RuntimeError("Approval is missing current authorization evidence or frozen recipe identity")
        matching = [run for run in approval.get("runs", [])
                    if run.get("variant") == args.variant and type(run.get("seed")) is int
                    and run["seed"] == args.seed and absolute(run.get("output", "")) == args.output]
        if len(matching) != 1:
            raise RuntimeError("Approval must name this exact variant, seed and new output directory once")
    return recipe, freeze, approval


def training_records(path):
    if path.name != "manifest_train.json":
        raise RuntimeError("Use the isolated manifest_train.json; the full/evaluation manifest is sealed")
    manifest = read_json(path)
    if manifest.get("schema") != "sample-domain-angle-v1":
        raise RuntimeError("Wrong SAMPLE manifest schema")
    classes = manifest["classes"]
    if len(classes) != 10 or classes != sorted(set(classes)):
        raise RuntimeError("Require ten unique, sorted classes")
    rows = manifest["records"]
    counts = Counter(row.get("split") for row in rows)
    if counts != {"train": TRAIN_COUNT}:
        raise RuntimeError(f"The training manifest must contain only806 train records: {counts}")
    # This manifest is source-only, so no real path, label, image, or pixel is
    # loaded even as JSON metadata. The sibling full/eval manifests stay sealed.
    train = [dict(row) for row in rows]
    del rows, manifest
    identifiers, paths, class_ids = set(), set(), set()
    for row in train:
        if row["domain"] != "synth" or type(row["elevation"]) is not int or row["elevation"] not in (14, 15, 16):
            raise RuntimeError("Every training row must be synthetic elevation14/15/16")
        label = row["label"]
        if type(label) is not int or label not in range(10) or row["class_name"] != classes[label]:
            raise RuntimeError("Training label/class mapping differs")
        if row["id"] in identifiers:
            raise RuntimeError("Duplicate training record id")
        source = absolute(row["path"])
        if source in paths or row["width"] != SIZE or row["height"] != SIZE:
            raise RuntimeError("Training paths must be unique and images must be native128x128")
        identifiers.add(row["id"])
        paths.add(source)
        class_ids.add(label)
    if class_ids != set(range(10)):
        raise RuntimeError("Training data must contain all ten classes")
    return classes, train


@contextlib.contextmanager
def gpu_locks():
    import fcntl
    handles = []
    try:
        # Always take the legacy lock, even when its file has not yet been made.
        # Do not unlink lock files: that would create separate locking inodes.
        if not LEGACY_LOCK.parent.is_dir():
            raise RuntimeError("Legacy GPU lock directory is missing")
        new_lock = HERE / "locks" / (GPU_UUID + ".lock")
        new_lock.parent.mkdir(parents=True, exist_ok=True)
        for path in (LEGACY_LOCK, new_lock):
            handle = path.open("a+")
            handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        for handle in reversed(handles):
            handle.close()


def assert_idle_gpu():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True, timeout=20,
    )
    busy = [line for line in result.stdout.splitlines() if line.strip().split(",")[0].strip() == GPU_UUID]
    if busy:
        raise RuntimeError(f"Assigned GPU has existing compute processes; no process is interrupted: {busy}")


def import_local(name):
    path = HERE / (name + ".py")
    spec = importlib.util.spec_from_file_location("sample_frozen_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ContinuousPermutation:
    def __init__(self, torch, size, seed):
        self.torch, self.size = torch, size
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed + 48151623)
        self.permutation, self.position, self.epoch = [], 0, -1
        self.consumed = 0
        self.chain = "0" * 64

    def next_batch(self, records):
        indices, segments = [], []
        before = self.consumed
        while len(indices) < BATCH:
            if self.position == len(self.permutation):
                self.permutation = self.torch.randperm(self.size, generator=self.generator).tolist()
                self.position = 0
                self.epoch += 1
                if sorted(self.permutation) != list(range(self.size)):
                    raise RuntimeError("Permutation is not a complete source pass")
            take = min(BATCH - len(indices), self.size - self.position)
            indices.extend(self.permutation[self.position:self.position + take])
            segments.append({"epoch": self.epoch, "offset": self.position, "count": take,
                             "permutation_sha256": digest(self.permutation)})
            self.position += take
        self.consumed += len(indices)
        payload = {"consumed_before": before, "consumed_after": self.consumed,
                   "record_ids": [records[index]["id"] for index in indices], "segments": segments}
        batch_hash = digest(payload)
        self.chain = digest({"previous": self.chain, "batch_sha256": batch_hash})
        return indices, {**payload, "batch_sha256": batch_hash, "stream_sha256": self.chain}

    def state_dict(self):
        return {"generator_state": self.generator.get_state(), "permutation": self.permutation,
                "position": self.position, "epoch": self.epoch, "consumed": self.consumed, "chain": self.chain}


def audit_sampler_boundary(torch, records, seed):
    probe = ContinuousPermutation(torch, len(records), seed)
    all_indices, batches = [], []
    for number in range(1, 25):
        indices, audit = probe.next_batch(records)
        if len(indices) != BATCH:
            raise RuntimeError("Index-only sampler audit found a non-full batch")
        all_indices.extend(indices)
        batches.append({"batch": number, **audit})
    if sorted(all_indices[:TRAIN_COUNT]) != list(range(TRAIN_COUNT)):
        raise RuntimeError("Sampler dropped or repeated a record within its first permutation")
    crossing = batches[22]["segments"]
    if len(crossing) != 2 or [part["count"] for part in crossing] != [14, 22]:
        raise RuntimeError("Batch23 does not concatenate the14-record tail and22-record next prefix")
    if probe.consumed != 24 * BATCH or len(set(all_indices[TRAIN_COUNT:])) != 24 * BATCH - TRAIN_COUNT:
        raise RuntimeError("Sampler boundary consumption is inconsistent")
    return {"status": "PASS", "index_only_batches": 24, "images_opened": 0,
            "model_calls": 0, "optimizer_updates": 0, "separate_rng_from_training": True,
            "first_permutation_complete_once": True, "crossing_batch": 23,
            "crossing_counts": [14, 22], "batches": batches}


def load_training_pixels(base, records):
    tensors, labels, opened = [], [], []
    for row in records:
        # The only image open in this executable is inside this train-only loop.
        path = absolute(row["path"])
        if sha(path) != row["byte_sha256"]:
            raise RuntimeError(f"Training image bytes changed: {row['id']}")
        with base.Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
            if (rgb.width, rgb.height) != (row["width"], row["height"]):
                raise RuntimeError("Training image dimensions differ from audited dimensions")
            pixel_hash = hashlib.sha256(canonical({"mode": "RGB", "width": rgb.width, "height": rgb.height}) + rgb.tobytes()).hexdigest()
            if pixel_hash != row["pixel_sha256"]:
                raise RuntimeError(f"Training decoded RGB pixels changed: {row['id']}")
            rgb = rgb.resize((SIZE, SIZE), base.Image.Resampling.BILINEAR)
            tensor = base.image_tensor(base.np.asarray(rgb).copy())
        if tuple(tensor.shape) != (3, SIZE, SIZE) or tensor.dtype != base.torch.float32 or not bool(tensor.isfinite().all()):
            raise RuntimeError("Invalid training tensor")
        tensors.append(tensor)
        labels.append(row["label"])
        opened.append({"id": row["id"], "byte_sha256": row["byte_sha256"], "pixel_sha256": pixel_hash})
    return base.torch.stack(tensors), base.torch.tensor(labels, dtype=base.torch.long), opened


def optimizer_groups(base, model):
    nn = base.nn
    norm_types = (nn.modules.batchnorm._BatchNorm, nn.modules.instancenorm._InstanceNorm,
                  nn.LayerNorm, nn.GroupNorm, nn.LocalResponseNorm)
    normalization, biases = set(), set()
    for module_name, module in model.named_modules():
        is_norm = isinstance(module, norm_types) or "norm" in type(module).__name__.lower()
        for name, parameter in module.named_parameters(recurse=False):
            if is_norm:
                normalization.add(id(parameter))
            if name == "bias" or name.endswith("_bias"):
                biases.add(id(parameter))
    groups = {"decay": [], "no_decay": []}
    ledger = {"decay": [], "no_decay": []}
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise RuntimeError("Parameter assigned more than once")
        seen.add(id(parameter))
        exempt = id(parameter) in normalization or id(parameter) in biases or name.endswith("bias")
        group = "no_decay" if exempt else "decay"
        groups[group].append(parameter)
        ledger[group].append({"name": name, "numel": parameter.numel(),
                              "reason": "normalization" if id(parameter) in normalization else "bias" if exempt else "other"})
    if seen != {id(p) for p in model.parameters() if p.requires_grad}:
        raise RuntimeError("Optimizer does not cover each trainable parameter exactly once")
    params = [{"params": groups[name], "weight_decay": 0.0 if name == "no_decay" else 0.0005,
               "group_name": name} for name in ("decay", "no_decay")]
    report = {name: {"weight_decay": 0.0 if name == "no_decay" else 0.0005,
                     "numel": sum(item["numel"] for item in rows), "parameters": rows}
              for name, rows in ledger.items()}
    return params, report


def finite_gradients(torch, model):
    missing, gradients = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                missing.append(name)
            else:
                gradients.append(parameter.grad.detach())
    if missing or not gradients:
        raise RuntimeError(f"Disconnected trainable parameters: {missing}")
    squared = torch.stack([value.double().square().sum() for value in gradients]).sum()
    norm = float(squared.sqrt())
    if not math.isfinite(norm):
        raise RuntimeError("Nonfinite gradient; optimizer update refused")
    return norm


def immutable_controls(model):
    return {name: value.detach().clone() for name, value in model.named_buffers()
            if name.startswith("backbone.p3.") and name.rsplit(".", 1)[-1] in
            ("lambda_s", "alpha", "beta_i", "rho_i", "eta_i")}


def checkpoint(torch, path, payload):
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_locked(args, recipe, freeze, approval, classes, records):
    assert_idle_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path.insert(0, str(HERE))
    import torch
    torch.set_num_threads(args.cpu_threads)
    device_audit = import_local("cuda_uuid_compat").ensure_uuid_cuda(GPU_UUID)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    factory = import_local("model_factory")
    base = factory.load_runtime(args.source_root)
    args.output.mkdir(parents=True, exist_ok=False)
    identity = {"schema": "sample-run-v1", "mode": args.mode, "variant": args.variant, "seed": args.seed,
                "freeze_sha256": sha(args.freeze_file), "protocol_sha256": freeze["protocol_sha256"],
                "manifest_sha256": freeze["manifest_sha256"], "audit_sha256": freeze["audit_sha256"],
                "optimizer_updates": args.probe_updates if args.mode == "probe" else UPDATES,
                "output": str(args.output), "gpu_uuid": GPU_UUID,
                "approval_sha256": sha(args.approval_file) if approval is not None else None}
    identity_hash = digest(identity)
    atomic_json(args.output / "config.json", {**identity, "identity_sha256": identity_hash,
                "protocol": recipe, "device": device_audit, "hostname": socket.gethostname(),
                "python": sys.version, "torch": torch.__version__, "started_utc": utc(),
                "test_image_opens": 0, "validation_or_test_inference": False})
    model = factory.create_model(base, "classification", 10, args.variant, args.seed)
    initial = factory.tensor_hashes(model)
    reference = factory.create_model(base, "classification", 10, "original", args.seed)
    reference_hashes = factory.tensor_hashes(reference)
    del reference
    common = [name for name in reference_hashes if not name.startswith("backbone.p3.")]
    if any(initial.get(name) != reference_hashes[name] for name in common):
        raise RuntimeError("Same-seed common backbone/head initialization differs from Original")
    atomic_json(args.output / "initialization.json", {
        "tensor_sha256": initial, "original_tensor_sha256": reference_hashes,
        "common_tensor_names": common, "common_tensor_sha256": digest({name: initial[name] for name in common}),
        "common_tensor_comparison": "PASS", "controls": factory.control_audit(base, model, args.variant),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    model = model.to(device="cuda:0", dtype=torch.float32)
    grouped, group_report = optimizer_groups(base, model)
    optimizer = torch.optim.SGD(grouped, lr=0.001, momentum=0.9, dampening=0.0,
                                nesterov=False, weight_decay=0.0)
    atomic_json(args.output / "optimizer_groups.json", {"optimizer": "SGD", "lr": 0.001,
        "momentum": 0.9, "nesterov": False, "groups": group_report,
        "physical_batch_size": BATCH, "gradient_accumulation": 1,
        "planned_optimizer_updates": identity["optimizer_updates"]})
    pixels, labels, opened = load_training_pixels(base, records)
    atomic_json(args.output / "training_input_audit.json", {"records": opened, "training_image_opens": len(opened),
        "test_image_opens": 0, "train_tensor_shape": list(pixels.shape), "classes": classes,
        "training_tensor_sha256": hashlib.sha256(pixels.numpy().tobytes()).hexdigest(),
        "normalization": recipe["input_transform"], "augmentation": "none"})
    pixels, labels = pixels.pin_memory(), labels.pin_memory()
    stream = ContinuousPermutation(torch, len(records), args.seed)
    atomic_json(args.output / "sampler_audit.json", audit_sampler_boundary(torch, records, args.seed))
    frozen_buffers = immutable_controls(model)
    base.seed_everything(args.seed + 1307002)
    model.train()
    timings, last_inputs, first_gradient = [], None, None
    started = time.monotonic()
    with (args.output / "updates.jsonl").open("x", encoding="utf-8") as ledger:
        for step in range(1, identity["optimizer_updates"] + 1):
            indices, stream_audit = stream.next_batch(records)
            if len(indices) != BATCH or stream.consumed != step * BATCH:
                raise RuntimeError("Continuous sampler violated physical batch/update accounting")
            torch.cuda.synchronize()
            tick = time.monotonic()
            inputs = pixels[indices].to("cuda:0", non_blocking=True)
            targets = labels[indices].to("cuda:0", non_blocking=True)
            if tuple(inputs.shape) != (BATCH, 3, SIZE, SIZE):
                raise RuntimeError("Physical batch is not exactly36x3x128x128")
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits, targets, label_smoothing=0.0)
            if logits.dtype != torch.float32 or not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-FP32 or nonfinite training loss")
            loss.backward()
            gradient_l2 = finite_gradients(torch, model)
            if step == 1:
                first_gradient = factory.gradient_audit(model)
            optimizer.step()
            for name, value in model.named_buffers():
                if name in frozen_buffers and not torch.equal(value, frozen_buffers[name]):
                    raise RuntimeError(f"Immutable control buffer changed: {name}")
            controls = factory.control_audit(base, model, args.variant)
            rates = [group["lr"] for group in optimizer.param_groups]
            decay = [group["weight_decay"] for group in optimizer.param_groups]
            if rates != [0.001, 0.001] or decay != [0.0005, 0.0]:
                raise RuntimeError("Actual optimizer learning rate/decay changed")
            torch.cuda.synchronize()
            seconds = time.monotonic() - tick
            timings.append(seconds)
            row = {"update": step, "physical_batch_size": BATCH, "optimizer_steps": step,
                   "gradient_accumulation": 1, "lr": rates, "weight_decay": decay,
                   "loss": float(loss.detach()), "gradient_l2": gradient_l2,
                   "update_seconds": seconds, "controls": controls, **stream_audit}
            ledger.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            ledger.flush()
            if step == 1 or step % 100 == 0 or step == identity["optimizer_updates"]:
                os.fsync(ledger.fileno())
                atomic_json(args.output / "status.json", {"status": "running", "mode": args.mode,
                    "update": step, "optimizer_steps": step, "loss": row["loss"],
                    "consumed_train_records": stream.consumed, "updated_utc": utc()})
            last_inputs = inputs.detach()
    if args.mode == "train" and stream.consumed != UPDATES * BATCH:
        raise RuntimeError("Formal training did not complete exactly10000 updates")
    output_checkpoint = args.output / ("probe_last.pt" if args.mode == "probe" else "last.pt")
    final_tensor_hashes = factory.tensor_hashes(model)
    checkpoint(torch, output_checkpoint, {"schema": "sample-final-checkpoint-v1", "identity": identity,
        "identity_sha256": identity_hash, "optimizer_updates": identity["optimizer_updates"],
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "stream": stream.state_dict(),
        "classes": classes, "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        "python_rng": random.getstate(), "numpy_rng": base.np.random.get_state(),
        "selection": "last_update_only", "training_only": True})
    # Reload check uses an already-loaded synthetic train batch. No file or
    # target-domain batch is opened; eval mode freezes BN and disables dropout.
    model.eval()
    with torch.no_grad():
        before_reload = model(last_inputs).detach()
    saved = torch.load(output_checkpoint, map_location="cpu", weights_only=False)
    if saved["identity_sha256"] != identity_hash or saved["optimizer_updates"] != identity["optimizer_updates"]:
        raise RuntimeError("Final checkpoint identity/update mismatch")
    model.load_state_dict(saved["model"], strict=True)
    if factory.tensor_hashes(model) != final_tensor_hashes:
        raise RuntimeError("Final checkpoint state differs after reload")
    with torch.no_grad():
        after_reload = model(last_inputs).detach()
    if not torch.equal(before_reload, after_reload) or not bool(torch.isfinite(after_reload).all()):
        raise RuntimeError("Reloaded final checkpoint train-batch logits are not exactly equal/finite")
    controls = factory.control_audit(base, model, args.variant)
    measured = timings[recipe["probe"]["timing_warmup_updates"]:]
    result = {"schema": "sample-training-result-v1", "status": "PASS", **identity,
        "identity_sha256": identity_hash, "finished_utc": utc(), "elapsed_seconds": time.monotonic() - started,
        "optimizer_steps": identity["optimizer_updates"], "consumed_training_records": stream.consumed,
        "final_stream_sha256": stream.chain, "final_checkpoint": str(output_checkpoint),
        "final_checkpoint_sha256": sha(output_checkpoint), "selection": "last_update_only",
        "test_image_opens": 0, "test_inference": False, "validation_used": False,
        "first_gradient_audit": first_gradient, "reloaded_controls": controls,
        "final_checkpoint_state_comparison": "PASS", "train_batch_reload_logits_exact": True,
        "timing": {"warmup_updates_excluded": recipe["probe"]["timing_warmup_updates"],
            "measured_updates": len(measured), "mean_update_seconds": sum(measured) / len(measured),
            "images_per_second": BATCH * len(measured) / sum(measured),
            "measurement": "training updates including transfer, forward, backward, SGD, and gradient/control checks; ledger IO excluded"}}
    atomic_json(args.output / "result.json", result)
    artifact_names = ["config.json", "initialization.json", "optimizer_groups.json", "training_input_audit.json", "sampler_audit.json",
                      "updates.jsonl", "result.json", output_checkpoint.name]
    marker = "probe_complete.json" if args.mode == "probe" else "training_complete.json"
    atomic_json(args.output / marker, {"status": "PASS", "mode": args.mode, "identity_sha256": identity_hash,
        "optimizer_updates": identity["optimizer_updates"], "artifacts": [
            {"path": str(args.output / name), "sha256": sha(args.output / name)} for name in artifact_names]})
    atomic_json(args.output / "status.json", {"status": "complete", "mode": args.mode,
        "optimizer_steps": identity["optimizer_updates"], "completion_marker": marker, "finished_utc": utc()})
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "train"), required=True)
    parser.add_argument("--manifest", type=absolute, required=True)
    parser.add_argument("--source-root", type=absolute, required=True)
    parser.add_argument("--freeze-file", type=absolute, required=True)
    parser.add_argument("--output", type=absolute, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-updates", type=int, default=16)
    parser.add_argument("--approval-file", type=absolute)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.seed < 0 or args.cpu_threads < 1 or args.cpu_threads > 16:
        parser.error("seed must be nonnegative; cpu-threads must be1..16")
    if not 12 <= args.probe_updates <= 16:
        parser.error("probe-updates must be12..16; formal training always uses10000 updates")
    if sys.platform != "linux":
        raise RuntimeError("Designated Linux CUDA server only; no local CPU/GPU fallback")
    if args.manifest.name != "manifest_train.json":
        raise ValueError("--manifest must point to source-only manifest_train.json, never full/eval JSON")
    if args.output.exists():
        raise FileExistsError("Output must be new; never overwrite a probe or an old trial")
    if HERE not in args.output.parents or args.manifest.parent == args.output:
        raise ValueError("Use a new output strictly inside the independent SAMPLE task directory")
    recipe, freeze, approval = verify_inputs(args)
    classes, records = training_records(args.manifest)
    with gpu_locks():
        try:
            run_locked(args, recipe, freeze, approval, classes, records)
        except BaseException as error:
            if args.output.is_dir():
                atomic_json(args.output / "failure.json", {"status": "FAIL", "mode": args.mode,
                    "error": f"{type(error).__name__}: {error}", "finished_utc": utc()})
            raise


if __name__ == "__main__":
    main()
