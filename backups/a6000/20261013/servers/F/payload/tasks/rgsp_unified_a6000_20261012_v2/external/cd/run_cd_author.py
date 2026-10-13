"""Per-pair, image-only pseudo-label training of hash-locked author CD models.

Run only on an explicitly selected registered server. Ground truth is opened
only after the selected checkpoint has completed inference. This is a new
common-optimization / pseudo-validation protocol, not a paper-score replay.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from author_models import (ROOT, RUNNABLE, build_criterion, build_model, extract_logits,
                           get_method_spec, native_preclassification, sha256)


def utcnow():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def logical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def gpu_alone(gpu_uuid, initialized=False):
    check = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=15)
    pids = {int(parts[1].strip()) for line in check.stdout.splitlines()
            for parts in [line.split(",")] if len(parts) == 2 and parts[0].strip() == gpu_uuid}
    if pids - {os.getpid()} or (initialized and os.getpid() not in pids):
        raise RuntimeError(f"Assigned GPU is not exclusively this runner's process: {sorted(pids)}")


def initialize_assigned_gpu(args):
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Execute only on the assigned Linux A6000 server")
    if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", args.gpu_uuid):
        raise ValueError("A full GPU UUID is required")
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    if previous is not None and previous != args.gpu_uuid:
        raise RuntimeError("Worker GPU mask disagrees with --gpu-uuid; refusing to expand visibility")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    gpu_alone(args.gpu_uuid)
    if args.cuda_compat:
        compat_path = Path(args.cuda_compat).resolve()
        spec = importlib.util.spec_from_file_location("cd_cuda_uuid_compat", compat_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        identity = module.ensure_uuid_cuda(args.gpu_uuid)
        identity["compatibility_source_sha256"] = sha256(compat_path)
    else:
        resolved = subprocess.run(["nvidia-smi", "--id=" + args.gpu_uuid,
            "--query-gpu=uuid,index,name", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15)
        rows = [line.strip().split(",") for line in resolved.stdout.splitlines() if line.strip()]
        if len(rows) != 1 or rows[0][0].strip() != args.gpu_uuid:
            raise RuntimeError("GPU UUID did not resolve to exactly the requested physical card")
        if int(torch._C._cuda_getDeviceCount()) != 1 or int(torch.cuda.device_count()) != 1:
            raise RuntimeError("CUDA UUID visibility/count mismatch; pass the audited --cuda-compat helper")
        identity = {"physical_gpu_uuid": args.gpu_uuid, "physical_gpu_index": int(rows[0][1]),
                    "physical_gpu_name": rows[0][2].strip(), "compatibility_applied": False}
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Single assigned CUDA device required; no CPU fallback")
    torch.cuda.set_device(0)
    allocation = torch.empty(1, device="cuda:0")
    torch.cuda.synchronize()
    gpu_alone(args.gpu_uuid, initialized=True)
    del allocation
    return identity


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def image_array(path):
    with Image.open(path) as image:
        values = np.asarray(image)
    if values.ndim == 3:
        # Exact source convention is first channel, not RGB luminance conversion.
        values = values[:, :, 0]
    if values.ndim != 2:
        raise ValueError(f"Unsupported image shape {values.shape}: {path}")
    return values.astype(np.float32)


class PatchDataset(Dataset):
    def __init__(self, features, indices, patch_size, labels=None):
        self.height, self.width = features.shape[:2]
        self.patch_size = patch_size
        self.indices = np.asarray(indices, dtype=np.int64)
        radius = patch_size // 2
        self.padded = np.pad(features, ((radius, radius), (radius, radius), (0, 0)), mode="constant")
        self.labels = None if labels is None else labels.reshape(-1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        center = int(self.indices[index])
        row, col = divmod(center, self.width)
        size = self.patch_size
        patch = self.padded[row:row + size, col:col + size]
        tensor = torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1)))
        if self.labels is None:
            return tensor
        target = int(self.labels[center] - 1)
        if target not in (0, 1):
            raise RuntimeError("Uncertain pseudo-label accidentally selected for supervision")
        return tensor, target


def pseudo_split(labels, native_quota, split_seed, val_fraction):
    rng = np.random.default_rng(split_seed)
    train, val, counts = [], [], []
    for class_index, quota in enumerate(native_quota):
        pool = np.flatnonzero(labels.reshape(-1) == class_index + 1)
        rng.shuffle(pool)
        selected = pool[:min(int(quota), len(pool))]
        if len(selected) < 2:
            raise RuntimeError(f"Insufficient reliable pseudo class {class_index}; no GT-assisted fallback")
        n_val = min(len(selected) - 1, max(1, int(round(len(selected) * val_fraction))))
        val.extend(selected[:n_val].tolist())
        train.extend(selected[n_val:].tolist())
        counts.append({"class": class_index, "available": len(pool), "nominal_quota": int(quota),
                       "selected": len(selected), "train": len(selected) - n_val, "val": n_val})
    train, val = np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64)
    rng.shuffle(train)
    rng.shuffle(val)
    if np.intersect1d(train, val).size:
        raise RuntimeError("Pseudo train/val centers overlap")
    return train, val, counts


def state_copy(model):
    # Keep every floating checkpoint tensor FP32; do not serialize notebook globals.
    return {key: value.detach().cpu().float().clone() if value.is_floating_point()
            else value.detach().cpu().clone() for key, value in model.state_dict().items()}


def checkpoint(path, model, optimizer, epoch, updates, val_loss, protocol_hash):
    temporary = path.with_name(path.name + ".tmp")
    torch.save({"model": state_copy(model), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "optimizer_updates": updates, "val_pseudo_loss": val_loss,
                "protocol_sha256": protocol_hash, "source_audit": model._cd_source_audit}, temporary)
    os.replace(temporary, path)


def validation(model, criterion, loader, method):
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to("cuda:0", non_blocking=True), labels.to("cuda:0", non_blocking=True)
            logits = extract_logits(method, model(data))
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite pseudo-validation loss")
            n = len(labels)
            loss_sum += float(loss.item()) * n
            correct += int((logits.argmax(1) == labels).sum().item())
            count += n
    return {"loss": loss_sum / count, "accuracy": correct / count, "count": count}


def scheduled_lr(update, total_updates, warmup_updates, base_lr):
    if update < warmup_updates:
        return base_lr * (update + 1) / max(1, warmup_updates)
    # Match the main worker's one-based update schedule; final update is 1%.
    progress = (update + 1 - warmup_updates) / max(1, total_updates - warmup_updates)
    return base_lr * (1.0 - 0.99 * min(1.0, max(0.0, progress)))


def confusion_metrics(target, prediction):
    gt, pred = target.astype(bool), prediction.astype(bool)
    tp, fp = int((gt & pred).sum()), int((~gt & pred).sum())
    fn, tn = int((gt & ~pred).sum()), int((~gt & ~pred).sum())
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    pcc = (tp + tn) / n
    expected = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (n * n)
    kappa = (pcc - expected) / (1 - expected) if expected < 1 else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "pixels": n,
            "precision": precision, "recall": recall, "f1": f1, "iou": iou,
            "overall_accuracy": pcc, "kappa": kappa, "percent_error": (fp + fn) / n * 100}


def final_inference(model, features, labels, patch_size, method, batch, workers):
    from skimage.measure import label as connected_labels
    uncertain = np.flatnonzero(labels.reshape(-1) == 1.5)
    prediction = (labels == 2).astype(np.uint8)
    dataset = PatchDataset(features, uncertain, patch_size)
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    offset = 0
    model.eval()
    with torch.no_grad():
        for data in loader:
            logits = extract_logits(method, model(data.to("cuda:0", non_blocking=True)))
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite final inference logits")
            values = logits.argmax(1).cpu().numpy().astype(np.uint8)
            prediction.reshape(-1)[uncertain[offset:offset + len(values)]] = values
            offset += len(values)
    if offset != len(uncertain):
        raise RuntimeError("Incomplete uncertain-pixel inference")
    raw = prediction.copy()
    components = connected_labels(prediction, connectivity=2)
    sizes = np.bincount(components.reshape(-1))
    remove = sizes <= 20
    remove[0] = False
    prediction[remove[components]] = 0
    return raw, prediction, {"neural_pixels": len(uncertain), "all_pixels": labels.size,
                             "fixed_component_threshold": 20, "connectivity": 2}


def verify_lant_hook_equivalence(out):
    """Tiny A6000-only value/gradient equivalence check; no training data or GT."""
    torch.manual_seed(641)
    original = build_model("lantnet", deduplicate_lant_hooks=False)
    adapted = build_model("lantnet", deduplicate_lant_hooks=True)
    adapted.load_state_dict(original.state_dict())
    original.train()
    adapted.train()
    records = []
    for repetition in range(2):
        data = torch.randn(2, 3, 7, 7, device="cuda:0")
        target = torch.tensor([0, 1], device="cuda:0")
        original.zero_grad(set_to_none=True)
        adapted.zero_grad(set_to_none=True)
        first, second = original(data), adapted(data)
        torch.nn.functional.cross_entropy(first, target).backward()
        torch.nn.functional.cross_entropy(second, target).backward()
        output_error = float((first - second).abs().max().item())
        gradient_error = 0.0
        for (name1, one), (name2, two) in zip(original.named_parameters(), adapted.named_parameters()):
            if name1 != name2 or (one.grad is None) != (two.grad is None):
                raise RuntimeError("LANTNet gradient topology changed")
            if one.grad is not None:
                gradient_error = max(gradient_error, float((one.grad - two.grad).abs().max().item()))
        if output_error > 1e-6 or gradient_error > 1e-6:
            raise RuntimeError(f"LANTNet hook equivalence failed: {output_error}, {gradient_error}")
        records.append({"repetition": repetition, "max_output_difference": output_error,
                        "max_gradient_difference": gradient_error})
    result = {"status": "PASS", "records": records, "adapter_sha256": sha256(ROOT / "author_models.py"),
              "source_audit": adapted._cd_source_audit, "timestamp": utcnow()}
    save_json(out / "lant_hook_equivalence.json", result)
    del original, adapted
    torch.cuda.empty_cache()
    return result


def execute(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # The worker creates this directory and opens run.log before exec. Preserve
    # actual earlier attempts: only pre-existing plain log files are acceptable.
    if any(not path.is_file() or path.suffix != ".log" for path in out.iterdir()):
        raise FileExistsError("Existing non-log experiment output; use a new worker attempt directory")
    started = time.perf_counter()
    try:
        identity = initialize_assigned_gpu(args)
        torch.set_num_threads(args.cpu_threads)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.method == "lantnet":
            verify_lant_hook_equivalence(out)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        spec = get_method_spec(args.method)
        preprocessing_start = time.perf_counter()
        first, second = image_array(args.image1), image_array(args.image2)
        features, pseudo, preprocessing = native_preclassification(args.method, first, second)
        train_indices, val_indices, sampling = pseudo_split(
            pseudo, spec["native_quota"], args.split_seed, args.val_fraction)
        preprocessing_seconds = time.perf_counter() - preprocessing_start
        np.savez_compressed(out / "image_only_protocol_arrays.npz", pseudo=pseudo,
                            train_indices=train_indices, val_indices=val_indices)
        source_manifest = {relative: sha256(ROOT / relative) for relative in
                           ("author_models.py", "run_cd_author.py", "source_registry.json")}
        protocol = {"protocol": "author_architecture_native_pseudo_common_optimization_v1",
            "method": args.method, "dataset": args.dataset,
            "split_seed": args.split_seed,
            "image1": {"sha256": sha256(args.image1)},
            "image2": {"sha256": sha256(args.image2)},
            "image_shape": list(first.shape), "input_channels": ["T1", "T2", "native_SRAD_logratio_DI"],
            "patch_size": spec["patch_size"], "batch_size": 16, "gradient_accumulation": 4,
            "nominal_effective_batch": 64, "partial_final_accumulation": "actual_sample_count_normalization",
            "optimizer": "AdamW", "lr": 0.0002, "weight_decay": 0.0005,
            "optimizer_parameter_groups": "bias and norm zero decay; ordinary weights 0.0005",
            "optimizer_betas": [0.937, 0.999], "optimizer_epsilon": 1e-8,
            "max_epochs": args.max_epochs, "patience": 30, "warmup_epochs": 3,
            "scheduler": "optimizer-update-wise linear warmup, linear decay to 0.01*base_lr",
            "precision": "FP32", "amp": False, "tf32": False, "initialization": "scratch",
            "checkpoint_selection": "minimum held-out image-only pseudo-label validation loss",
            "early_stop": "30 consecutive non-improving pseudo-validation epochs; min_delta=0",
            "ground_truth_access_before_final_inference": False, "transductive": True,
            "ground_truth_spatial_split": False, "original_paper_reproduction": False,
            "pseudo_validation_fraction": args.val_fraction, "sampling": sampling,
            "sampling_adaptation": "Deterministic source nominal quotas, truncate to available real samples; fix source off-by-one/dummy-zero sampling. Hold out pseudo centers only; image patches can overlap.",
            "postprocessing": "native uncertain-only classification; fixed <=20 connected-component removal",
            "preprocessing": preprocessing, "source_manifest": source_manifest,
            "protocol_arrays_logical_sha256": logical_digest({
                "pseudo": hashlib.sha256(pseudo.astype("<f4").tobytes()).hexdigest(),
                "train_indices": hashlib.sha256(train_indices.astype("<i8").tobytes()).hexdigest(),
                "val_indices": hashlib.sha256(val_indices.astype("<i8").tobytes()).hexdigest()})}
        protocol_hash = logical_digest(protocol)
        save_json(out / "protocol.json", {"protocol_sha256": protocol_hash, "protocol": protocol})
        save_json(out / "run_metadata.json", {"seed": args.seed, "mode": args.mode,
            "started_at": utcnow(), "preprocessing_seconds": preprocessing_seconds,
            "protocol_arrays_file_sha256": sha256(out / "image_only_protocol_arrays.npz"),
            "image1_path": str(Path(args.image1).resolve()), "image2_path": str(Path(args.image2).resolve()),
            "ground_truth_path_not_read": str(Path(args.ground_truth).resolve()),
            "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0), "logical_device": 0,
                        "host": os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME"),
                        "command": sys.argv, "identity": identity}})
        train_loader = DataLoader(PatchDataset(features, train_indices, spec["patch_size"], pseudo),
            batch_size=16, shuffle=True, num_workers=args.workers, pin_memory=True,
            generator=torch.Generator().manual_seed(args.seed), drop_last=False)
        model = build_model(args.method, allow_fp32_adaptation=True)
        if args.method == 'lantnet' and args.mode == 'probe':
            original = build_model('lantnet', deduplicate_lant_hooks=False)
            adapted = build_model('lantnet', deduplicate_lant_hooks=True)
            adapted.load_state_dict(original.state_dict(), strict=True)
            checks = []
            for index in range(3):
                inputs = torch.randn(4, 3, 7, 7, device='cuda')
                outputs, gradients = [], []
                for candidate in (original, adapted):
                    candidate.zero_grad(set_to_none=True)
                    torch.manual_seed(20261012 + index)
                    logits = extract_logits('lantnet', candidate(inputs))
                    logits.square().sum().backward()
                    outputs.append(logits.detach().clone())
                    gradients.append({name: None if p.grad is None else p.grad.detach().clone()
                                      for name, p in candidate.named_parameters()})
                if not torch.equal(outputs[0], outputs[1]):
                    raise RuntimeError('LANTNet hook adaptation changed forward values')
                for name, gradient in gradients[0].items():
                    other = gradients[1][name]
                    if (gradient is None) != (other is None) or (gradient is not None and not torch.equal(gradient, other)):
                        raise RuntimeError('LANTNet hook adaptation changed gradients: ' + name)
                checks.append({'forward_equal': True, 'all_gradients_equal': True})
            save_json(out / 'lant_hook_equivalence.json', {'status': 'PASS', 'checks': checks,
                      'ground_truth_used': False, 'hardware': args.gpu_uuid})
            del original, adapted, outputs, gradients
            torch.cuda.empty_cache()
        criterion = build_criterion(args.method)
        save_json(out / "model_source_audit.json", model._cd_source_audit)
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from run_cross import optimizer_for
        optimizer = optimizer_for(torch, model)
        steps_per_epoch = math.ceil(len(train_loader) / 4)
        total_updates, warmup_updates = args.max_epochs * steps_per_epoch, 3 * steps_per_epoch
        if args.mode == "probe":
            # Do not construct a validation or test dataset/loader in this branch.
            model.train()
            iterator = iter(train_loader)
            durations, probe_losses = [], []
            for update_index in range(6):
                torch.cuda.synchronize()
                update_started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                sample_count, weighted_loss = 0, 0.0
                for micro in range(4):
                    try:
                        data, labels = next(iterator)
                    except StopIteration:
                        iterator = iter(train_loader)
                        data, labels = next(iterator)
                    data, labels = data.to("cuda:0", non_blocking=True), labels.to("cuda:0", non_blocking=True)
                    logits = extract_logits(args.method, model(data))
                    loss = criterion(logits, labels)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite probe training loss")
                    n = len(labels)
                    (loss * n).backward()
                    sample_count += n
                    weighted_loss += float(loss.item()) * n
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(sample_count)
                        if not torch.isfinite(parameter.grad).all():
                            raise RuntimeError("Non-finite probe gradient")
                for group in optimizer.param_groups:
                    group["lr"] = scheduled_lr(update_index, total_updates, warmup_updates, 0.0002)
                optimizer.step()
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - update_started)
                probe_losses.append(weighted_loss / sample_count)
            mean_update = sum(durations[2:]) / 4
            # Validation/inference overhead is an explicit arithmetic estimate
            # from train time, not a measurement or access to held-out data.
            per_forward_micro_estimate = mean_update / 12
            validation_estimate = math.ceil(len(val_indices) / 16) * args.max_epochs * per_forward_micro_estimate
            inference_estimate = math.ceil(int((pseudo == 1.5).sum()) / 16) * per_forward_micro_estimate
            estimated = preprocessing_seconds + mean_update * total_updates + validation_estimate + inference_estimate
            result = {"status": "PROBE_PASS", "task": "change_detection", "method": args.method,
                "dataset": args.dataset, "seed": args.seed, "device": "cuda:0", "gpu_uuid": args.gpu_uuid,
                "test_evaluated": False, "ground_truth_read": False, "validation_loader_created": False,
                "test_loader_created": False, "optimizer_updates": 6, "warmup_probe_updates": 2,
                "timed_updates": 4, "update_seconds": durations, "training_losses": probe_losses,
                "mean_timed_update_seconds": mean_update, "preprocessing_seconds": preprocessing_seconds,
                "estimated_full_seconds": estimated, "planned_max_updates": total_updates,
                "estimate_components": {"train_seconds": mean_update * total_updates,
                    "unmeasured_pseudo_validation_seconds": validation_estimate,
                    "unmeasured_uncertain_inference_seconds": inference_estimate},
                "estimate_caveat": "6-update extrapolation; pseudo-validation and uncertain inference use train-time/3 forward estimate, never sampled",
                "protocol_sha256": protocol_hash, "source_audit": model._cd_source_audit,
                "timestamp": utcnow()}
            gpu_alone(args.gpu_uuid, initialized=True)
            save_json(out / "result.json", result)
            save_json(out / "status.json", result)
            print(json.dumps(result, allow_nan=False), flush=True)
            return result
        val_loader = DataLoader(PatchDataset(features, val_indices, spec["patch_size"], pseudo),
            batch_size=16, shuffle=False, num_workers=args.workers, pin_memory=True)
        best, best_epoch, stale, updates = math.inf, 0, 0, 0
        epoch_count = args.max_epochs
        for epoch in range(1, epoch_count + 1):
            model.train()
            epoch_start = time.perf_counter()
            loss_sum, count, chunk_samples = 0.0, 0, 0
            optimizer.zero_grad(set_to_none=True)
            for step, (data, labels) in enumerate(train_loader):
                data, labels = data.to("cuda:0", non_blocking=True), labels.to("cuda:0", non_blocking=True)
                logits = extract_logits(args.method, model(data))
                loss = criterion(logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite train loss epoch={epoch} step={step}")
                n = len(labels)
                (loss * n).backward()
                loss_sum += float(loss.item()) * n
                count += n
                chunk_samples += n
                flush = (step + 1) % 4 == 0 or step + 1 == len(train_loader)
                if flush:
                    lr = scheduled_lr(updates, total_updates, warmup_updates, 0.0002)
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(chunk_samples)
                            if not torch.isfinite(parameter.grad).all():
                                raise RuntimeError("Non-finite gradient; no skipped/AMP step")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    chunk_samples = 0
                    updates += 1
            validation_result = validation(model, criterion, val_loader, args.method)
            improved = validation_result["loss"] < best
            if improved:
                best, best_epoch, stale = validation_result["loss"], epoch, 0
                checkpoint(out / "best.pt", model, optimizer, epoch, updates, best, protocol_hash)
            else:
                stale += 1
            checkpoint(out / "last.pt", model, optimizer, epoch, updates,
                       validation_result["loss"], protocol_hash)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - epoch_start
            projected_hours = elapsed * args.max_epochs / 3600
            ledger = {"epoch": epoch, "optimizer_updates": updates, "training_examples": count,
                      "train_loss": loss_sum / count, "pseudo_validation": validation_result,
                      "best_epoch": best_epoch, "best_pseudo_val_loss": best,
                      "lr_after_epoch": optimizer.param_groups[0]["lr"], "epoch_seconds": elapsed,
                      "projected_max_epoch_hours": projected_hours, "ground_truth_read": False,
                      "timestamp": utcnow()}
            with (out / "epochs.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(ledger, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            save_json(out / "status.json", {"status": "TRAINING", **ledger})
            print(json.dumps(ledger), flush=True)
            if stale >= 30:
                break
        restored = torch.load(out / "best.pt", map_location="cpu")
        if restored["protocol_sha256"] != protocol_hash:
            raise RuntimeError("Checkpoint protocol mismatch")
        for name, tensor in restored["model"].items():
            if tensor.is_floating_point() and (tensor.dtype != torch.float32 or not torch.isfinite(tensor).all()):
                raise RuntimeError(f"Invalid FP32 checkpoint tensor: {name}")
        model.load_state_dict(restored["model"], strict=True)
        model.to("cuda:0")
        reloaded_val = validation(model, criterion, val_loader, args.method)
        if abs(reloaded_val["loss"] - best) > 1e-5 * max(1.0, abs(best)):
            raise RuntimeError("Best checkpoint pseudo-validation reload mismatch")
        raw_prediction, prediction, inference = final_inference(
            model, features, pseudo, spec["patch_size"], args.method, 16, args.workers)
        Image.fromarray(raw_prediction * 255).save(out / "prediction_before_postprocess.png")
        Image.fromarray(prediction * 255).save(out / "prediction.png")
        # This is intentionally the first GT read. No metric can influence
        # pseudo labels, training, early stopping, threshold or postprocessing.
        gt_raw = image_array(args.ground_truth)
        if gt_raw.shape != prediction.shape or not np.isfinite(gt_raw).all():
            raise RuntimeError("Ground truth shape/finiteness mismatch after final inference")
        unique = np.unique(gt_raw)
        if not set(unique).issubset({0.0, 1.0, 255.0}):
            raise RuntimeError(f"Unsupported GT encoding {unique.tolist()}; no inferred threshold")
        gt = gt_raw > 0
        duration = time.perf_counter() - started
        result = {"status": "TRAINING_COMPLETE", "task": "change_detection", "method": args.method, "dataset": args.dataset,
            "seed": args.seed, "protocol": protocol["protocol"], "protocol_sha256": protocol_hash,
            "device": "cuda:0", "gpu_uuid": args.gpu_uuid, "test_evaluated": True,
            "best_epoch": best_epoch, "trained_epochs": epoch, "optimizer_updates": updates,
            "best_pseudo_val_loss": best, "reloaded_pseudo_validation": reloaded_val,
            "metrics": confusion_metrics(gt, prediction),
            "before_fixed_postprocess_metrics": confusion_metrics(gt, raw_prediction),
            "inference": inference, "ground_truth_sha256": sha256(args.ground_truth),
            "ground_truth_first_access": "after selected-checkpoint inference completed",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "best_weight_sha256": sha256(out / "best.pt"), "last_weight_sha256": sha256(out / "last.pt"),
            "checkpoint": {"path": "best.pt", "sha256": sha256(out / "best.pt")},
            "elapsed_seconds": duration, "seed_count_recommendation_from_actual_runtime":
                3 if duration <= 3600 else 2 if duration <= 18000 else 1,
            "original_paper_score_reproduction": False, "completed_at": utcnow()}
        save_json(out / "result.json", result)
        outputs = ["protocol.json", "model_source_audit.json", "image_only_protocol_arrays.npz",
                   "epochs.jsonl", "best.pt", "last.pt", "prediction.png",
                   "prediction_before_postprocess.png", "result.json"]
        save_json(out / "completion.json", {"status": "TRAINING_COMPLETE", "method": args.method,
            "result_sha256": sha256(out / "result.json"),
            "protocol_sha256": protocol_hash, "source_manifest": source_manifest,
            "outputs": {name: sha256(out / name) for name in outputs}, "completed_at": utcnow()})
        save_json(out / "status.json", {"status": "TRAINING_COMPLETE", "best_epoch": best_epoch,
                                        "trained_epochs": epoch, "completed_at": utcnow()})
        print(json.dumps(result, allow_nan=False), flush=True)
        return result
    except Exception as error:
        save_json(out / "failure.json", {"status": "FAILED", "error_type": type(error).__name__,
            "error": str(error), "timestamp": utcnow(), "elapsed_seconds": time.perf_counter() - started})
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=RUNNABLE, required=True)
    parser.add_argument("--mode", choices=("probe", "train"), required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--cuda-compat", help="Audited existing cuda_uuid_compat.py; loaded without running its CLI")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--image1", required=True)
    parser.add_argument("--image2", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output", required=True, help="Fresh run directory; an existing worker log alone is allowed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=20261012)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=150)
    args = parser.parse_args()
    if args.max_epochs != 150 or args.val_fraction != 0.2 or args.workers < 0 or args.cpu_threads < 1:
        parser.error("Frozen production protocol requires max_epochs=150, val_fraction=0.2, nonnegative workers")
    if args.seed not in (0, 1, 2):
        parser.error("Only authorized seeds 0,1,2 are supported")
    return args


if __name__ == "__main__":
    execute(parse_args())
