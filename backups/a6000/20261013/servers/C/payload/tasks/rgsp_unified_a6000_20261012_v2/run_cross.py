#!/usr/bin/env python3
"""Independent, CUDA-only unified classification/supervised-CD experiment.

The old campaign supplies frozen networks/transforms/metric functions only.
Its training CLI, seed barrier and result directories are never invoked.
The lane worker owns the physical GPU locks. No installation or CPU fallback.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VARIANTS = {"Original": "original", "Matched075": "matched_scale075", "Full": "full"}
HYPERPARAMETERS = {
    "physical_batch": 16, "gradient_accumulation": 4, "nominal_effective_batch": 64,
    "epochs": 150, "patience": 30, "optimizer": "AdamW", "lr0": 0.0002,
    "betas": [0.937, 0.999], "epsilon": 1e-8,
    "weight_decay": {"weights": 0.0005, "bias_and_norm": 0.0},
    "lr_schedule": "linear warmup 3 actual epochs; linear decay to lr0*0.01 at planned epoch150",
    "gradient_clip_l2": 10.0, "module_warmup_epochs": 0,
    "tail_policy": "flush every epoch, normalize accumulated gradients by actual sample count",
    "batch_norm": "physical microbatch statistics, not physical batch64",
    "precision": "FP32; AMP/TF32 disabled", "initialization": "scratch",
    "ema": {"decay": 0.9999, "tau_updates": 2000},
    "workers": 0, "normalization": "fixed (RGB/255 - 0.5)/0.5, no test statistics",
    "test_policy": "one final evaluation of earliest best validation EMA checkpoint",
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value, replace=False):
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError("Refusing to overwrite artifact: " + str(path))
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def learning_rate(update_number, updates_per_epoch):
    warmup, total = 3 * updates_per_epoch, 150 * updates_per_epoch
    if not 1 <= update_number <= total:
        raise ValueError("Update outside the frozen training budget")
    if update_number <= warmup:
        return 0.0002 * update_number / warmup
    return 0.0002 * (1.0 - 0.99 * (update_number - warmup) / (total - warmup))


def gpu_pids(gpu_uuid):
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                             "--format=csv,noheader,nounits"], capture_output=True,
                            text=True, check=True, timeout=15)
    return {int(parts[1].strip()) for line in result.stdout.splitlines()
            for parts in [line.split(",")] if len(parts) == 2 and parts[0].strip() == gpu_uuid}


def require_gpu_alone(gpu_uuid, initialized=False):
    pids = gpu_pids(gpu_uuid)
    if pids - {os.getpid()} or (initialized and os.getpid() not in pids):
        raise RuntimeError("GPU not exclusively available; no takeover: " + repr(sorted(pids)))


def file_inventory(args):
    files = [Path(args.campaign_root) / name for name in
             ("cross_trial.py", "cross_data.py", "manifest.py", "cuda_uuid_compat.py")]
    files.extend(path for path in Path(args.source_root).rglob("*")
                 if path.is_file() and path.suffix in (".py", ".yaml", ".yml")
                 and "__pycache__" not in path.parts)
    files.extend([Path(__file__), ROOT / "external/classification/factory.py"])
    files.extend(ROOT.joinpath("data").glob("*.py"))
    files.extend([ROOT / "data/source_registry.json", ROOT / "detection/covered_flops.py"])
    return [{"path": str(path.resolve()), "sha256": sha(path)} for path in sorted(set(files))]


def verify_data(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("task") == "classification":
        sys.path.insert(0, str(ROOT / "data"))
        from prepare_classification import verify_classification_manifest
        verified = verify_classification_manifest(path)
    elif manifest.get("task") == "change_detection":
        sys.path.insert(0, str(ROOT / "data"))
        from prepare_cd import verify_cd_manifest
        verified = verify_cd_manifest(path)
    else:
        raise ValueError("Only classification and supervised change_detection are supported")
    bound = manifest.get("rgsp_data", {})
    if not re.fullmatch(r"[0-9a-f]{64}", bound.get("logical_manifest_sha256", "")):
        raise ValueError("Missing verified logical data fingerprint")
    if not all(any(row["split"] == split for row in manifest["records"])
               for split in ("train", "val", "test")):
        raise ValueError("A nonempty independent train/val/test split is required")
    return manifest, verified


def optimizer_for(torch, model):
    norms = tuple(cls for name, cls in vars(torch.nn).items()
                  if isinstance(cls, type) and "Norm" in name)
    groups, seen = {"decay": [], "no_decay": []}, set()
    for _, module in model.named_modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            groups["no_decay" if "bias" in name or isinstance(module, norms) else "decay"].append(param)
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if seen != expected:
        raise RuntimeError("Optimizer parameter coverage mismatch")
    return torch.optim.AdamW([
        {"params": groups["decay"], "weight_decay": 5e-4},
        {"params": groups["no_decay"], "weight_decay": 0.0},
    ], lr=2e-4, betas=(0.937, 0.999), eps=1e-8)


def update_ema(torch, ema, model, updates):
    decay = 0.9999 * (1.0 - math.exp(-updates / 2000.0))
    with torch.no_grad():
        current = model.state_dict()
        for name, value in ema.state_dict().items():
            source = current[name].detach()
            if value.is_floating_point():
                value.mul_(decay).add_(source, alpha=1 - decay)
            else:
                value.copy_(source)


def basic_gradient_audit(torch, model):
    missing, bad = [], []
    total = 0.0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            missing.append(name)
        elif not bool(torch.isfinite(p.grad).all()):
            bad.append(name)
        else:
            total += float(p.grad.detach().square().sum())
    if missing or bad or not math.isfinite(total) or total <= 0:
        raise RuntimeError("Invalid/disconnected trainable gradients: " + repr((missing, bad, total)))
    return {"missing": missing, "nonfinite": bad, "gradient_l2": math.sqrt(total)}


def checkpoint(torch, path, payload):
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    if temporary.exists():
        raise FileExistsError(temporary)
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def profile(torch, model, task, size, output):
    """Post-training only; reported FLOPs are explicitly covered operations."""
    inputs = tuple(torch.zeros((1, 3, size, size), device="cuda")
                   for _ in range(2 if task == "change_detection" else 1))
    model.eval()
    report = {"params": sum(p.numel() for p in model.parameters()),
              "input_shape_each": [1, 3, size, size], "input_tensors": len(inputs),
              "precision": "FP32", "batch": 1, "hardware": "RTX A6000",
              "covered_flops": None, "complete_operation_coverage": False}
    try:
        with torch.no_grad():
            for _ in range(5):
                model(*inputs)
            torch.cuda.synchronize()
            durations = []
            for _ in range(20):
                start = time.perf_counter()
                model(*inputs)
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - start)
            report["latency_ms_mean"] = sum(durations) * 1000 / len(durations)
            report["throughput_samples_per_second"] = 1000 / report["latency_ms_mean"]
            report["latency_scope"] = "model-only; synthetic shape-matched inputs; no I/O/pre/postprocessing"
        from detection.covered_flops import profile_model
        covered = profile_model(model, inputs=inputs)
        report["covered_operation_profile"] = covered
        if covered["status"] == "PASS":
            report["covered_flops"] = covered["covered_flops_g"] * 1e9
        report["profiler"] = "THOP legacy x2 + actual selective-scan/einsum/cross-merge covered arithmetic"
        report["status"] = "PARTIAL_COVERAGE_MEASURED" if covered["status"] == "PASS" else "PROFILE_FAILED_PERFORMANCE_RESULTS_PRESERVED"
    except Exception as error:
        report["status"] = "PROFILE_FAILED_PERFORMANCE_RESULTS_PRESERVED"
        report["error"] = type(error).__name__ + ": " + str(error)
    write_json(output / "complexity.json", report)
    return report


def run(args):
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Execute only on the assigned Linux A6000 server")
    if args.seed not in (0, 1, 2):
        raise ValueError("Only seeds 0/1/2 are authorized for this new campaign")
    output = Path(args.output).resolve()
    if not output.is_absolute() or len(output.parts) < 6:
        raise ValueError("Use an explicit independent experiment output")
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("protocol.json", "result.json", "best.pt", "failure.json")):
        raise FileExistsError("This attempt already has artifacts; preserve it and use a new attempt directory")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    require_gpu_alone(args.gpu_uuid)
    manifest, data_verified = verify_data(args.manifest)
    task = manifest["task"]
    if manifest["dataset"] != args.dataset:
        raise ValueError("Dataset identity mismatch")
    if task == "change_detection" and args.method not in VARIANTS:
        raise ValueError("External patch methods require their own unsupervised runner")
    classes = len(manifest.get("classes", [])) if task == "classification" else 1
    size = 224 if task == "classification" else 192
    counts = {split: sum(row["split"] == split for row in manifest["records"])
              for split in ("train", "val", "test")}
    updates_per_epoch = math.ceil(math.ceil(counts["train"] / 16) / 4)
    source_files = file_inventory(args)
    source_logical = [{"file": str(Path(row["path"]).name), "sha256": row["sha256"]} for row in source_files]
    protocol = {"schema": "rgsp-unified-cross-20261012-v1", "task": task, "dataset": args.dataset,
                "method": args.method, "hyperparameters": HYPERPARAMETERS, "input_size": size,
                "classes": manifest.get("classes", ["change"]), "counts": counts,
                "data_protocol": manifest.get("protocol"), "data_sha256": manifest["rgsp_data"]["logical_manifest_sha256"],
                "source_sha256": digest(source_logical), "updates_per_epoch": updates_per_epoch,
                "selection": "validation macro_f1" if task == "classification" else "validation native unique-pixel f1, threshold grid0.10:0.05:0.90",
                "loss": "cross_entropy label_smoothing=0.1" if task == "classification" else
                        "train-crop-only positive-weight BCE + per-physical-microbatch soft Dice, sample-weighted accumulation",
                "supervision": "ground-truth supervised; not comparable with transductive unsupervised CD literature",
                "augmentation": "paired h/v flips + right-angle rotations; classification random crop after resize1.12x",
                "source_reused": "frozen networks, transforms and metric functions only; old campaign training never invoked",
                "literature_performance_reuse": False}
    protocol_sha = digest(protocol)
    write_json(output / "protocol.json", {"protocol_sha256": protocol_sha, "protocol": protocol})
    write_json(output / "source_inventory.json", source_files)
    write_json(output / "data_verified.json", data_verified)
    sys.path.insert(0, str(Path(args.campaign_root).resolve()))
    cross = load_module("unified_frozen_cross", Path(args.campaign_root) / "cross_trial.py")
    compatibility = load_module("unified_cuda_uuid", Path(args.campaign_root) / "cuda_uuid_compat.py")
    import torch
    identity = compatibility.ensure_uuid_cuda(args.gpu_uuid)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    base = cross.load_runtime(args.source_root)
    base.seed_everything(args.seed)
    if args.method in VARIANTS:
        model = cross.create_model(base, task, classes, VARIANTS[args.method], args.seed)
        control = cross.control_audit(base, model, VARIANTS[args.method])
    else:
        factory = load_module("unified_classification_factory", ROOT / "external/classification/factory.py")
        model = factory.create_model(args.method, classes, size)
        control = getattr(model, "_rgsp_external_metadata", {})
    init_hash = cross.tensor_hashes(model)
    model = model.float().to("cuda:0")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError("A non-FP32 parameter violates the fixed protocol")
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = optimizer_for(torch, model)
    dataset_type = cross.dataset_class(base)
    train = dataset_type(manifest, "train", size, args.seed)
    # Probe never creates a validation/test loader or evaluates a holdout split.
    val = dataset_type(manifest, "val", size, args.seed) if args.mode == "train" else None
    criterion = base.build_loss(task, manifest, output, torch.device("cuda:0"), label_smoothing=0.1)
    write_json(output / "initialization.json", {"state_sha256": digest(init_hash), "tensors": init_hash, "control": control})
    identity.update({"started_utc": utc(), "pid": os.getpid(), "seed": args.seed,
                     "mode": args.mode, "protocol_sha256": protocol_sha, "test_evaluated": False})
    write_json(output / "runtime.json", identity)
    require_gpu_alone(args.gpu_uuid, initialized=True)
    updates = 0
    best_score, best_epoch, best_threshold, stale = -math.inf, 0, None, 0
    timing, grad_evidence = [], None
    training_start = time.perf_counter()
    final_epoch = 0
    for epoch in range(1, 151):
        require_gpu_alone(args.gpu_uuid, initialized=True)
        model.train()
        loader = cross.make_loader(base, train, 16, 0, epoch, args.seed, True)
        optimizer.zero_grad(set_to_none=True)
        accumulated, epoch_samples, loss_sum, microsteps = 0, 0, 0.0, 0
        start_update = time.perf_counter()
        epoch_start = time.perf_counter()
        for batch_index, packed in enumerate(loader):
            inputs, targets = base.move_batch(packed, torch.device("cuda:0"))
            n = int(targets.shape[0])
            predictions = model(*inputs)
            loss = criterion(predictions, targets)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite loss at epoch/batch " + str((epoch, batch_index)))
            (loss * n).backward()
            accumulated += n
            epoch_samples += n
            microsteps += 1
            loss_sum += float(loss.detach()) * n
            if microsteps % 4 == 0 or batch_index + 1 == len(loader):
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated)
                if grad_evidence is None:
                    grad_evidence = (cross.gradient_audit(model) if args.method in VARIANTS else
                                     basic_gradient_audit(torch, model))
                    write_json(output / "gradient_audit.json", grad_evidence)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
                lr = learning_rate(updates + 1, updates_per_epoch)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
                updates += 1
                update_ema(torch, ema, model, updates)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start_update
                timing.append(elapsed)
                append_json(output / "updates.jsonl", {"epoch": epoch, "update": updates,
                            "samples": accumulated, "lr": lr, "gradient_norm": float(norm), "seconds": elapsed})
                accumulated = 0
                start_update = time.perf_counter()
                if args.mode == "probe" and updates == 6:
                    measured = timing[2:]
                    per_update = sum(measured) / len(measured)
                    train_estimate = per_update * updates_per_epoch * 150
                    # Validation cost estimate uses only split sizes, never val/test inference.
                    factor = 1.0 + 0.4 * counts["val"] / counts["train"]
                    estimate = train_estimate * factor + 60
                    result = {"status": "PROBE_PASS", "seed": args.seed, "task": task,
                              "dataset": args.dataset, "method": args.method,
                              "estimated_full_seconds": estimate, "estimate_basis": "4 timed optimizer updates after 2 warmup updates; full150epoch estimate plus size-only evaluation allowance",
                              "timed_update_seconds": measured, "updates_per_epoch": updates_per_epoch,
                              "params": sum(p.numel() for p in model.parameters()),
                              "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                              "test_evaluated": False, "validation_evaluated": False,
                              "protocol_sha256": protocol_sha}
                    write_json(output / "result.json", result)
                    print(json.dumps(result), flush=True)
                    return
        if epoch_samples != counts["train"] or accumulated or microsteps != len(loader):
            raise RuntimeError("Training epoch dropped, duplicated or failed to flush samples")
        if args.mode == "probe":
            continue
        if args.method in VARIANTS:
            cross.control_audit(base, model, VARIANTS[args.method])
        val_metrics, threshold = cross.evaluate(base, ema, val, 16, 0, torch.device("cuda:0"), args.seed, classes)
        score = val_metrics["macro_f1" if task == "classification" else "f1"]
        if not math.isfinite(score):
            raise RuntimeError("Nonfinite validation score")
        improved = score > best_score
        if improved:
            best_score, best_epoch, best_threshold, stale = score, epoch, threshold, 0
            state = {key: value.detach().cpu().clone() for key, value in ema.state_dict().items()}
            checkpoint(torch, output / "best.pt", {"state_dict": state, "epoch": epoch,
                       "seed": args.seed, "protocol_sha256": protocol_sha,
                       "validation_metrics": val_metrics, "threshold": threshold,
                       "precision": "FP32", "checkpoint_selection": "EMA validation only"})
        else:
            stale += 1
        final_epoch = epoch
        record = {"epoch": epoch, "optimizer_updates": updates, "samples": epoch_samples,
                  "microbatches": microsteps, "train_loss": loss_sum / epoch_samples,
                  "lr": lr, "validation": val_metrics, "threshold": threshold,
                  "best_epoch": best_epoch, "best_score": best_score,
                  "epoch_seconds": time.perf_counter() - epoch_start, "timestamp_utc": utc()}
        append_json(output / "epochs.jsonl", record)
        write_json(output / "progress.json", record, replace=True)
        print(json.dumps({"epoch": epoch, "updates": updates, "loss": record["train_loss"],
                          "val_score": score, "best_epoch": best_epoch}), flush=True)
        if stale >= 30:
            break
    training_seconds = time.perf_counter() - training_start
    verified_final = verify_data(args.manifest)[1]
    if verified_final != data_verified or file_inventory(args) != source_files:
        raise RuntimeError("Source or dataset changed during training; result not accepted")
    best = torch.load(output / "best.pt", map_location="cpu")
    if best["protocol_sha256"] != protocol_sha or best["seed"] != args.seed:
        raise RuntimeError("Checkpoint protocol/seed mismatch")
    if any(value.is_floating_point() and value.dtype != torch.float32 for value in best["state_dict"].values()):
        raise RuntimeError("Selected checkpoint lost FP32 precision")
    ema.load_state_dict(best["state_dict"], strict=True)
    if args.method in VARIANTS:
        cross.control_audit(base, ema, VARIANTS[args.method])
    test = dataset_type(manifest, "test", size, args.seed)
    test_metrics, test_threshold = cross.evaluate(base, ema, test, 16, 0, torch.device("cuda:0"), args.seed,
                                                 classes, threshold=best_threshold)
    write_json(output / "test.json", {"metrics": test_metrics, "threshold": test_threshold,
               "checkpoint_sha256": sha(output / "best.pt"), "test_evaluated_once": True})
    complexity = profile(torch, ema, task, size, output)
    metrics = {key: value for key, value in test_metrics.items() if not isinstance(value, str)}
    result = {"status": "TRAINING_COMPLETE", "seed": args.seed, "task": task, "dataset": args.dataset,
              "method": args.method, "metrics": metrics, "protocol_sha256": protocol_sha,
              "checkpoint": {"path": "best.pt", "sha256": sha(output / "best.pt")},
              "epochs_completed": final_epoch, "optimizer_updates": updates,
              "best_epoch": best_epoch, "best_validation_score": best_score,
              "training_seconds": training_seconds, "elapsed_seconds": time.perf_counter() - training_start,
              "threshold_from_validation": best_threshold, "test_evaluated": True,
              "params": complexity["params"], "complexity": complexity,
              "finished_utc": utc(), "protocol_group": "new_unified_supervised",
              "historical_results_reused": False}
    write_json(output / "result.json", result)
    write_json(output / "completion.json", {"status": "TRAINING_COMPLETE", "result_sha256": sha(output / "result.json")})
    print(json.dumps({"status": "TRAINING_COMPLETE", "result": str(output / "result.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--mode", choices=("probe", "train"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as error:
        output = Path(args.output)
        if output.is_dir() and not (output / "failure.json").exists():
            write_json(output / "failure.json", {"status": "FAILED", "utc": utc(),
                       "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
