#!/usr/bin/env python3
"""Server-only fixed-protocol detection probe/train; no resume or CPU fallback.

The controller owns all old/new GPU locks. This process checks only the exact
assigned UUID and that no foreign compute process appears. Source model, loss,
augmentation and metric implementations are imported from --source-root.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import os
from pathlib import Path
import random
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from types import SimpleNamespace

sys.dont_write_bytecode = True

from data_binding import prepare_data, recheck_data, write_json
from detector_spec import (METHODS, PROTOCOL, SCHEMA, digest, learning_rate,
                           method_name, model_config, sha, update_budget)


def utc():
    return datetime.now(timezone.utc).isoformat()


def compute_pids(gpu_uuid):
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                             "--format=csv,noheader,nounits"],
                            text=True, capture_output=True, check=True, timeout=15)
    pids = set()
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0] == gpu_uuid:
            pids.add(int(parts[1]))
    return pids


def assert_gpu(gpu_uuid, initialized=False):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid:
        raise RuntimeError("Assigned exact GPU UUID mask changed")
    pids = compute_pids(gpu_uuid)
    permitted = {os.getpid()} if initialized else {os.getpid()}
    if pids - permitted or (initialized and pids != permitted):
        raise RuntimeError("BLOCKED_GPU_OCCUPIED: no takeover; active PIDs=" + repr(sorted(pids)))
    return sorted(pids)


def cuda_identity(gpu_uuid, torch):
    result = subprocess.run(["nvidia-smi", "--id=" + gpu_uuid,
                             "--query-gpu=uuid,index,name", "--format=csv,noheader,nounits"],
                            text=True, capture_output=True, check=True, timeout=15)
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError("GPU UUID did not resolve to one device")
    parts = [part.strip() for part in rows[0].split(",")]
    if len(parts) != 3 or parts[0] != gpu_uuid:
        raise RuntimeError("Physical GPU identity mismatch")
    native, reported = int(torch._C._cuda_getDeviceCount()), int(torch.cuda.device_count())
    repaired = False
    if native != 1:
        raise RuntimeError("Exactly one CUDA runtime device required")
    if native != reported:
        if not torch.__version__.startswith("1.13.") or reported != 0:
            raise RuntimeError("Unrecognized CUDA device-count mismatch")
        torch.cuda.device_count = lambda: int(torch._C._cuda_getDeviceCount())
        repaired = True  # Process-local fix only; installed torch is unchanged.
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("No designated CUDA device; no CPU fallback")
    torch.cuda.set_device(0)
    return {"hostname": socket.gethostname(), "gpu_uuid": gpu_uuid, "gpu_index": int(parts[1]),
            "gpu_name": parts[2], "torch": torch.__version__, "cuda": torch.version.cuda,
            "uuid_count_compatibility_applied": repaired, "installation_modified": False}


def source_inventory(root):
    package = root / "ultralytics"
    required = [package / "nn/modules/mamba_yolo.py", package / "nn/tasks.py",
                package / "utils/loss.py", package / "models/yolo/detect/val.py",
                root / "tools/validate_rgsp_expanded.py"]
    if not all(path.is_file() for path in required):
        raise RuntimeError("Expected frozen project Ultralytics tree is incomplete")
    files = {path for path in package.rglob("*") if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}}
    files.update(required)
    return [{"path": str(path.relative_to(root)).replace("\\", "/"), "sha256": sha(path)}
            for path in sorted(files)]


def check_expected_source(root, path):
    if path is None:
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("code_hashes", value.get("files")) if isinstance(value, dict) else value
    if not isinstance(rows, list) or not rows:
        raise ValueError("Source manifest must contain path/sha256 rows")
    for row in rows:
        item = Path(row["path"])
        item = item if item.is_absolute() else root / item
        if not item.is_file() or sha(item) != row["sha256"]:
            raise RuntimeError("Frozen source-manifest mismatch: " + str(item))


def append_row(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def main(args):
    if sys.platform != "linux":
        raise RuntimeError("Linux server only; local execution and CPU fallback are prohibited")
    if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", args.gpu_uuid):
        raise ValueError("Full exact GPU UUID required")
    assert_gpu(args.gpu_uuid)
    args.source_root = args.source_root.resolve(strict=True)
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError("No overwrite or implicit resume: " + str(args.output))
    if args.output == args.source_root or args.source_root in args.output.parents:
        raise ValueError("Output must be outside the frozen source tree")
    args.output.mkdir(parents=True)
    args.output_created = True
    args.method = method_name(args.method)
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["COMET_MODE"] = "DISABLED"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["YOLO_CONFIG_DIR"] = str(args.output / "ultralytics_settings")
    os.environ["MPLCONFIGDIR"] = str(args.output / "matplotlib")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    check_expected_source(args.source_root, args.source_manifest)
    sources = source_inventory(args.source_root)
    runner_sources = {path.name: sha(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    write_json(args.output / "source_hashes.json", sources)
    write_json(args.output / "runner_hashes.json", runner_sources)
    data, records, data_audit = prepare_data(args)
    budget = update_budget(data_audit["counts_verified"]["train"])
    sys.path.insert(0, str(args.source_root))
    sys.path.insert(0, str(args.source_root / "tools"))
    import numpy as np
    import torch
    identity = cuda_identity(args.gpu_uuid, torch)  # Repair the known 1.13 UUID parser before framework imports.
    import ultralytics
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_yolo_dataset
    from ultralytics.data.build import seed_worker
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import ModelEMA
    from validate_rgsp_expanded import audit_controls
    if args.source_root not in Path(ultralytics.__file__).resolve().parents:
        raise RuntimeError("An installed package shadowed the supplied frozen source")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    if args.method in {"Original", "Matched075", "Full"}:
        import ultralytics.nn.modules.common_utils_mbyolo as common_mamba
        import ultralytics.nn.modules.mamba_yolo as mamba
        if getattr(common_mamba, "selective_scan_cuda_core", None) is None or not mamba.USE_CUDA_SCAN:
            raise RuntimeError("Compiled selective-scan backend missing; no reference/CPU fallback")

    def seed_all():
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    def tensor_hash(tensor):
        value = tensor.detach().cpu().contiguous()
        return digest({"dtype": str(value.dtype), "shape": list(value.shape),
                       "bytes_sha256": __import__("hashlib").sha256(value.numpy().tobytes()).hexdigest()})

    def model_state_digest(model):
        return digest({name: tensor_hash(value) for name, value in model.state_dict().items()})

    def verify_fp32(model):
        for name, value in model.state_dict().items():
            if value.is_floating_point() and (value.dtype != torch.float32 or not torch.isfinite(value).all()):
                raise FloatingPointError("Non-FP32 or nonfinite model state: " + name)

    def controls(model, optimizer=None):
        rows = audit_controls(model, optimizer)
        expected = 1 if args.method in {"Matched075", "Full"} else 0
        if len(rows) != expected:
            raise RuntimeError("Unexpected number of actual RGSP control blocks")
        for name, block in model.named_modules():
            if type(block).__name__ == "DS_VSSBlockSweep":
                if name != "model.3.0" and name != "model.3":
                    raise RuntimeError("RGSP block is not the single intended P3 block: " + name)
                if block.control != METHODS[args.method] or block.ds_ssm.state_dim != 16:
                    raise RuntimeError("RGSP control/state dimension differs from frozen configuration")
        return rows

    nc = data["nc"]
    config = model_config(args.method, nc)
    write_json(args.output / "model.yaml", config)
    aug = PROTOCOL["augmentation"]
    overrides = dict(task="detect", mode="train", imgsz=640, batch=16,
                     workers=args.workers, seed=args.seed, deterministic=True,
                     amp=False, half=False, cache=False, rect=False, fraction=1.0,
                     pretrained=False, resume=False, epochs=150, patience=30,
                     optimizer="AdamW", lr0=0.0002, lrf=0.01, momentum=0.937,
                     weight_decay=0.0005, nbs=64, warmup_epochs=0.0,
                     warmup_bias_lr=0.0, warmup_momentum=0.937, module_warmup_epochs=0,
                     cos_lr=False, lr_schedule="fixed", multi_scale=False,
                     hsv_h=aug["hsv_h"], hsv_s=aug["hsv_s"], hsv_v=aug["hsv_v"],
                     degrees=aug["degrees"], translate=aug["translate"], scale=aug["scale"],
                     shear=aug["shear"], perspective=aug["perspective"],
                     flipud=aug["flipud"], fliplr=aug["fliplr"], mosaic=aug["mosaic"],
                     mixup=0.0, copy_paste=0.0, close_mosaic=15,
                     box=7.5, cls=0.5, dfl=1.5, device="", conf=0.001, iou=0.7,
                     max_det=300, plots=False, save_json=False, save_txt=False,
                     project=str(args.output), name="run", exist_ok=True)
    cfg = get_cfg(overrides=overrides)
    seed_all()
    model = DetectionModel(config, ch=3, nc=nc, verbose=False)
    common_keys = None
    if args.method in {"Original", "Matched075", "Full"}:
        seed_all()
        reference = DetectionModel(model_config("Original", nc), ch=3, nc=nc, verbose=False)
        state, ref = model.state_dict(), reference.state_dict()
        common_keys = sorted(key for key in ref if not key.startswith("model.3."))
        if any(key not in state or state[key].shape != ref[key].shape for key in common_keys):
            raise RuntimeError("Mamba common-tensor alignment failed")
        for key in common_keys:
            state[key].copy_(ref[key])
        del state, ref, reference
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
    model.args, model.nc, model.names = cfg, nc, data["names"]
    model = model.float().cuda().train()
    verify_fp32(model)
    initial_controls = controls(model)
    initial_state_sha256 = model_state_digest(model)
    common_sha256 = (digest({key: tensor_hash(model.state_dict()[key]) for key in common_keys})
                     if common_keys is not None else None)
    groups, group_names = {"weights": [], "norm": [], "bias": []}, {"weights": [], "norm": [], "bias": []}
    norm_types = tuple(value for key, value in torch.nn.__dict__.items() if "Norm" in key and isinstance(value, type))
    seen_parameters = set()
    for module_name, module in model.named_modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad or id(parameter) in seen_parameters:
                continue
            seen_parameters.add(id(parameter))
            group = "bias" if "bias" in name else ("norm" if isinstance(module, norm_types) else "weights")
            groups[group].append(parameter)
            group_names[group].append(module_name + "." + name if module_name else name)
    if seen_parameters != {id(parameter) for parameter in model.parameters() if parameter.requires_grad}:
        raise RuntimeError("Optimizer parameter coverage mismatch")
    optimizer = torch.optim.AdamW(
        [{"params": parameters, "weight_decay": 0.0005 if name == "weights" else 0.0, "group_name": name}
         for name, parameters in groups.items() if parameters],
        lr=0.0002, betas=(0.937, 0.999), eps=1e-8)
    controls(model, optimizer)
    from covered_flops import profile_model
    complexity = profile_model(model, imgsz=640)
    write_json(args.output / "complexity.json", complexity)
    # Complexity availability is independent of a valid performance experiment.
    # Missing coverage stays null with its failure reason; it is never a fake 0.
    if complexity.get("status") != "PASS":
        print(json.dumps({"complexity_status": complexity.get("status"),
                          "note": "Incomplete FLOPs are disclosed separately; no substitute value"}), flush=True)
    if model_state_digest(model) != initial_state_sha256:
        raise RuntimeError("Complexity profiler changed the training model")
    seed_all()  # Same data-augmentation RNG starting point despite different model construction.
    loader_seed = 6148914691236517205 + args.seed
    generator = torch.Generator().manual_seed(loader_seed)

    def make_loader(split, training=False):
        dataset = build_yolo_dataset(cfg, data[split], 16, data,
                                     mode="train" if training else "val", rect=not training, stride=32)
        if len(dataset) != data_audit["counts_verified"][split]:
            raise RuntimeError("Runtime loader silently discarded data: " + split)
        return torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=training,
                    num_workers=args.workers if training else 0, pin_memory=True,
                    collate_fn=dataset.collate_fn, worker_init_fn=seed_worker,
                    generator=generator if training else torch.Generator().manual_seed(loader_seed),
                    drop_last=False, persistent_workers=False)

    train_loader = make_loader("train", training=True)
    if len(train_loader) != budget["microbatches_per_epoch"] or train_loader.batch_size != 16:
        raise RuntimeError("Physical batch or microbatch count differs from protocol")
    protocol = {**copy.deepcopy(PROTOCOL), "dataset_id": args.dataset_id, "method": args.method,
                "seed": args.seed, "budget": budget, "mode": args.mode,
                "source_sha256": digest(sources), "runner_sha256": digest(runner_sources),
                "model_config_sha256": digest(config), "data_audit_sha256": sha(args.output / "data_audit.json"),
                "data_manifest_sha256": data_audit.get("input_manifest", {}).get("sha256"),
                "verified_data_sha256": data_audit["actual_records_sha256"],
                "train_workers": args.workers, "loader_generator_seed": loader_seed,
                "determinism": "seeded workers and warn_only deterministic algorithms; custom CUDA kernels may not be bitwise deterministic",
                "probe_timing_warmup_updates": args.probe_warmup_updates if args.mode == "probe" else None,
                "probe_measured_updates": args.probe_updates if args.mode == "probe" else None}
    write_json(args.output / "protocol.json", protocol)
    protocol_sha256 = sha(args.output / "protocol.json")
    runtime = {"identity": identity, "source_file_count": len(sources), "ultralytics": ultralytics.__version__,
               "method": args.method, "initial_state_sha256": initial_state_sha256,
               "common_tensor_sha256": common_sha256, "initial_controls": initial_controls,
               "optimizer_groups": [{key: group[key] for key in ("group_name", "lr", "weight_decay", "betas", "eps")}
                                    for group in optimizer.param_groups],
               "named_optimizer_groups": group_names, "budget": budget,
               "parameters": sum(parameter.numel() for parameter in model.parameters()),
               "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
               "protocol_sha256": protocol_sha256, "actual_compute_pids": assert_gpu(args.gpu_uuid, True)}
    write_json(args.output / "runtime_audit.json", runtime)
    ema = ModelEMA(model)
    completed_updates = 0
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()

    def update_group(batches, update_number):
        lr = learning_rate(update_number, budget)
        for group in optimizer.param_groups:
            group["lr"] = lr
        sample_count, loss_items, seen_sizes = 0, [], []
        for batch in batches:
            count = int(batch["img"].shape[0])
            if not 1 <= count <= 16:
                raise RuntimeError("Microbatch changed from physical batch16")
            batch["img"] = batch["img"].cuda(non_blocking=True).float() / 255.0
            loss, items = model(batch)
            if not bool(torch.isfinite(loss).all()):
                raise FloatingPointError("Nonfinite detection loss")
            loss.backward()  # Frozen loss returns normalized microbatch objective * sample_count.
            sample_count += count
            seen_sizes.append(count)
            loss_items.append([float(value) for value in items.detach().cpu()])
        finite_checks = []
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(sample_count)
                finite_checks.append(torch.isfinite(parameter.grad).all())
        if not finite_checks or not bool(torch.stack(finite_checks).all()):
            raise FloatingPointError("Nonfinite or absent gradients")
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
        if not math.isfinite(norm):
            raise FloatingPointError("Nonfinite gradient norm")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ema.update(model)
        return {"update": update_number, "lr": lr, "samples": sample_count,
                "microbatch_sizes": seen_sizes, "gradient_norm_before_clip": norm,
                "loss_items": loss_items}

    def groups_from_loader():
        pending = []
        for batch in train_loader:
            pending.append(batch)
            if len(pending) == 4:
                yield pending
                pending = []
        if pending:
            yield pending

    if args.mode == "probe":
        measured, all_rows = [], []
        group_iterator = iter(groups_from_loader())
        for index in range(args.probe_warmup_updates + args.probe_updates):
            try:
                batches = next(group_iterator)
            except StopIteration:
                group_iterator = iter(groups_from_loader())
                batches = next(group_iterator)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            row = update_group(batches, index + 1)
            torch.cuda.synchronize()
            row.update(seconds=time.perf_counter() - begin, measured=index >= args.probe_warmup_updates)
            controls(model, optimizer)
            assert_gpu(args.gpu_uuid, True)
            append_row(args.output / "updates.jsonl", row)
            all_rows.append(row)
            if row["measured"]:
                measured.append(row)
            print(json.dumps({key: row[key] for key in ("update", "samples", "seconds", "measured")}), flush=True)
        verify_fp32(model)
        recheck_data(records, data_audit)
        if source_inventory(args.source_root) != sources:
            raise RuntimeError("Source code changed during probe")
        if {path.name: sha(path) for path in sorted(Path(__file__).parent.glob("*.py"))} != runner_sources:
            raise RuntimeError("Runner code changed during probe")
        mean_update = sum(row["seconds"] for row in measured) / len(measured)
        estimate = mean_update * budget["planned_total_updates"]
        result = {"status": "PROBE_PASS", "schema": SCHEMA, "method": args.method, "seed": args.seed,
                  "paper_eligible": False, "test_evaluated": False, "val_evaluated": False,
                  "protocol_sha256": protocol_sha256, "source_sha256": digest(sources),
                  "code_sha256": digest(runner_sources), "data_sha256": data_audit["actual_records_sha256"],
                  "measured_updates": len(measured), "timing_warmup_updates": args.probe_warmup_updates,
                  "mean_update_seconds": mean_update,
                  "estimated_full_seconds": estimate,
                  "estimate_scope": "planned150-epoch training steps only; excludes validation/checkpoint/I/O overhead and early stopping",
                  "images_per_second": sum(row["samples"] for row in measured) / sum(row["seconds"] for row in measured),
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                  "complexity": complexity, "controls": controls(model, optimizer), "finished_utc": utc()}
        write_json(args.output / "result.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return

    val_loader = make_loader("val")
    weights_dir = args.output / "weights"
    weights_dir.mkdir()
    best_path, last_path = weights_dir / "best.pt", weights_dir / "last.pt"
    best_fitness, best_epoch, stale_epochs = -math.inf, None, 0
    epoch_rows = []

    def validation(selected, loader, split, epoch):
        validation_cfg = copy.deepcopy(cfg)
        validation_cfg.mode, validation_cfg.split = "val", split
        validation_cfg.half, validation_cfg.save_json, validation_cfg.plots = False, False, False
        validator = DetectionValidator(loader, save_dir=args.output / "evaluation" / split,
                                      args=validation_cfg, _callbacks={"on_val_start": []})
        validator.callbacks = {}
        def label_loss_items(items, prefix="val"):
            return {prefix + "/" + name: float(value)
                    for name, value in zip(("box_loss", "cls_loss", "dfl_loss"), items)}
        context = SimpleNamespace(device=torch.device("cuda:0"), data=data, amp=False,
                    ema=SimpleNamespace(ema=selected), model=selected,
                    loss_items=torch.zeros(3, device="cuda:0"),
                    stopper=SimpleNamespace(possible_stop=False), epoch=epoch - 1, epochs=150,
                    label_loss_items=label_loss_items)
        validator(trainer=context)  # No AutoBackend fusion, FP16 cast, downloads, or integrations.
        box = validator.metrics.box
        values = {"P": float(box.mp), "R": float(box.mr), "AP50": float(box.map50),
                  "AP75": float(box.all_ap[:, 5].mean()) if len(box.all_ap) else 0.0,
                  "AP50_95": float(box.map)}
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values.values()):
            raise FloatingPointError("Invalid validation/test metrics")
        return values, {key: float(value) for key, value in validator.speed.items()}

    def cpu_tree(value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: cpu_tree(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cpu_tree(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cpu_tree(item) for item in value)
        return copy.deepcopy(value)

    def checkpoint_model(value):
        cloned = copy.deepcopy(value).float().cpu()
        if hasattr(cloned, "criterion"):
            delattr(cloned, "criterion")  # Recreate unregistered criterion device tensors on reload.
        return cloned

    def save_checkpoint(path, epoch, fitness, stop_reason):
        payload = {"schema": SCHEMA, "epoch": epoch - 1, "best_epoch": best_epoch,
                   "best_fitness": best_fitness, "fitness": fitness,
                   "model": checkpoint_model(model), "ema": checkpoint_model(ema.ema),
                   "updates": ema.updates, "completed_updates": completed_updates,
                   "optimizer": cpu_tree(optimizer.state_dict()), "train_args": vars(cfg),
                   "protocol_sha256": protocol_sha256, "source_sha256": digest(sources),
                   "code_sha256": digest(runner_sources), "data_sha256": data_audit["actual_records_sha256"],
                   "model_config": config, "stop_reason": stop_reason, "date": utc(),
                   "version": ultralytics.__version__, "resume_supported": False}
        payload["ema_state_sha256"] = model_state_digest(payload["ema"])
        verify_fp32(payload["model"])
        verify_fp32(payload["ema"])
        temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    for epoch in range(1, 151):
        begin_epoch = time.perf_counter()
        assert_gpu(args.gpu_uuid, True)
        model.train()
        if epoch == 136:
            train_loader.dataset.close_mosaic(hyp=cfg)
        samples_seen, microbatches_seen, epoch_updates, losses = 0, 0, [], []
        for batches in groups_from_loader():
            row = update_group(batches, completed_updates + 1)
            completed_updates += 1
            row["epoch"] = epoch
            samples_seen += row["samples"]
            microbatches_seen += len(row["microbatch_sizes"])
            losses.extend(row["loss_items"])
            epoch_updates.append(row)
            append_row(args.output / "updates.jsonl", row)
        if (samples_seen != data_audit["counts_verified"]["train"]
                or microbatches_seen != budget["microbatches_per_epoch"]
                or len(epoch_updates) != budget["updates_per_epoch"]
                or completed_updates != epoch * budget["updates_per_epoch"]):
            raise RuntimeError("Epoch sample/update ledger failed; no dropped tails permitted")
        ema.update_attr(model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])
        epoch_controls = controls(model, optimizer)
        val_values, val_speed = validation(ema.ema, val_loader, "val", epoch)
        fitness = 0.1 * val_values["AP50"] + 0.9 * val_values["AP50_95"]
        improved = fitness > best_fitness
        if improved:
            best_fitness, best_epoch, stale_epochs = fitness, epoch, 0
        else:
            stale_epochs += 1
        stop_reason = "patience30" if stale_epochs >= 30 else ("max_epochs150" if epoch == 150 else None)
        row = {"epoch": epoch, "samples": samples_seen, "microbatches": microbatches_seen,
               "optimizer_updates": completed_updates, "epoch_updates": len(epoch_updates),
               "last_group_samples": epoch_updates[-1]["samples"], "last_lr": epoch_updates[-1]["lr"],
               "train_loss_items_mean": np.asarray(losses).mean(axis=0).tolist(),
               "val_metrics": val_values, "fitness": fitness, "best_epoch": best_epoch,
               "stale_epochs": stale_epochs, "stop_reason": stop_reason,
               "controls": epoch_controls, "seconds_before_checkpoint": time.perf_counter() - begin_epoch}
        save_checkpoint(last_path, epoch, fitness, stop_reason)
        if improved:
            # Both files are task-owned; no historical checkpoint can be overwritten.
            with last_path.open("rb") as source, best_path.with_name("best.pt.tmp").open("wb") as target:
                import shutil
                shutil.copyfileobj(source, target)
                target.flush()
                os.fsync(target.fileno())
            os.replace(best_path.with_name("best.pt.tmp"), best_path)
        row["epoch_seconds"] = time.perf_counter() - begin_epoch
        append_row(args.output / "optimizer_ledger.jsonl", row)
        epoch_rows.append(row)
        csv_path = args.output / "results.csv"
        with csv_path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["epoch", "optimizer_updates", "fitness", "P", "R", "AP50", "AP75", "AP50_95"])
            if epoch == 1:
                writer.writeheader()
            writer.writerow({"epoch": epoch, "optimizer_updates": completed_updates, "fitness": fitness, **val_values})
        print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
        gc.collect()
        if stop_reason is not None:
            break
    if not epoch_rows or epoch_rows[-1]["stop_reason"] is None or not best_path.is_file():
        raise RuntimeError("Training is not terminal; refusing a completion result")
    best_hash = sha(best_path)
    import inspect
    trusted_load = {"weights_only": False} if "weights_only" in inspect.signature(torch.load).parameters else {}
    saved = torch.load(best_path, map_location="cpu", **trusted_load)  # Only this task-created, hash-bound checkpoint.
    if (saved["protocol_sha256"] != protocol_sha256 or saved["source_sha256"] != digest(sources)
            or saved["ema_state_sha256"] != model_state_digest(saved["ema"])):
        raise RuntimeError("Selected checkpoint binding/roundtrip mismatch")
    selected = saved["ema"]
    verify_fp32(selected)
    selected_controls = controls(selected)
    del model, ema, optimizer, saved
    gc.collect()
    torch.cuda.empty_cache()
    selected = selected.float().cuda()
    test_loader = make_loader("test")  # First test loader/model evaluation, after all selection is terminal.
    test_metrics, test_speed = validation(selected, test_loader, "test", epoch_rows[-1]["epoch"])
    if sha(best_path) != best_hash:
        raise RuntimeError("Best checkpoint changed during test")
    recheck_data(records, data_audit)
    if source_inventory(args.source_root) != sources:
        raise RuntimeError("Source code changed during formal training")
    if {path.name: sha(path) for path in sorted(Path(__file__).parent.glob("*.py"))} != runner_sources:
        raise RuntimeError("Runner code changed during formal training")
    result = {"status": "TRAINING_COMPLETE", "schema": SCHEMA, "method": args.method,
              "dataset_id": args.dataset_id, "seed": args.seed, "paper_eligible": True,
              "test_evaluated": True, "selection_split": "val", "test_split": "test",
              "protocol_sha256": protocol_sha256, "source_sha256": digest(sources),
              "code_sha256": digest(runner_sources), "data_sha256": data_audit["actual_records_sha256"],
              "dataset_manifest_sha256": data_audit.get("input_manifest", {}).get("sha256"),
              "checkpoint": {"path": str(best_path.resolve()), "sha256": best_hash},
              "checkpoint_fp32_verified": True, "checkpoint_controls": selected_controls,
              "epochs_completed": epoch_rows[-1]["epoch"], "best_epoch": best_epoch,
              "optimizer_updates": completed_updates, "stop_reason": epoch_rows[-1]["stop_reason"],
              "metrics": test_metrics, "metric_unit": "fraction_0_to_1",
              "metrics_percent": {key: value * 100 for key, value in test_metrics.items()},
              "complexity": complexity, "parameters": complexity["parameters"],
              "covered_flops_g": complexity["covered_flops_g"],
              "test_speed_ms": test_speed, "elapsed_seconds": time.perf_counter() - started,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
              "protocol_is_new_not_historical_reuse": True, "finished_utc": utc(),
              "gpu_identity": identity, "final_gpu_pids": assert_gpu(args.gpu_uuid, True)}
    write_json(args.output / "result.json", result)
    write_json(args.output / "completion.json", {"status": "TRAINING_COMPLETE", "schema": SCHEMA,
                "result_sha256": sha(args.output / "result.json"), "protocol_sha256": protocol_sha256,
                "checkpoint": result["checkpoint"], "finished_utc": utc()})
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--dataset-id", choices=["ssdd", "sadd", "sar_ship", "sar_aircraft"], required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--mode", choices=["probe", "train"], required=True)
    parser.add_argument("--workers", type=int, choices=[0, 1, 2, 4], default=4)
    parser.add_argument("--probe-warmup-updates", type=int, default=2)
    parser.add_argument("--probe-updates", type=int, default=4)
    args = parser.parse_args()
    if args.data is None and args.dataset_manifest is None:
        parser.error("--data or --dataset-manifest is required")
    if args.probe_warmup_updates < 2 or args.probe_updates < 4:
        parser.error("Probe requires at least2 timing warmup and4 measured optimizer updates")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    try:
        main(arguments)
    except BaseException as error:
        if getattr(arguments, "output_created", False):
            message = str(error)
            status = "BLOCKED_OOM" if "out of memory" in message.lower() else "FAILED"
            if message.startswith("BLOCKED_"):
                status = message.split(":", 1)[0]
            report = {"status": status, "method": arguments.method, "mode": arguments.mode,
                      "error_type": type(error).__name__, "error": message, "finished_utc": utc(),
                      "paper_eligible": False, "automatic_batch_reduction": False,
                      "cpu_fallback": False, "completion_written": False}
            write_json(arguments.output / "failure.json", report, replace=True)
            if not (arguments.output / "result.json").exists():
                write_json(arguments.output / "result.json", report)
        raise
