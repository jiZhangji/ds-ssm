#!/usr/bin/env python3
"""Finite, CUDA-only, seed-gated controlled classification/CD trial.

New experiment, not a relabeling of the old VMamba output-adapter or warmup runs.
Shares existing dataset transforms, loss functions and UPerNet-lite implementation;
uses a real full Mamba-YOLO-T P3/8 replacement and a live final P5 readout.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import socket
import sys
from pathlib import Path

import manifest as campaign
from cross_data import DATASETS, verify_staged

VARIANTS = ("original", "matched_scale075", "full")
SCHEMA = "rgsp-cross-trial-v1"


def load_runtime(source):
    source = Path(source).resolve()
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("rgsp_shared_v2_frozen", source / "tools/run_shared_sar_tasks_v2.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def create_model(base, task, classes, variant, seed):
    torch, nn = base.torch, base.nn

    class CrossModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.task = task
            # Construct the entire baseline first, identically in every arm.
            # A forked CPU RNG initializes only the substituted block: common
            # downstream modules AND task head therefore have identical tensors.
            self.backbone = base.MambaYOLOTBackbone(False, 16)
            if variant != "original":
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed + 1307001)
                    self.backbone.p3 = base.DS_VSSBlockSweep(64, 64, 16, control=variant)
            if task == "classification":
                self.head = nn.Sequential(nn.LayerNorm(256), nn.Dropout(0.2), nn.Linear(256, classes))
            else:
                self.decoder = base.UPerNetLiteDecoder((32, 64, 128, 256), 1, width=96)

        def forward(self, *inputs):
            first = self.backbone(inputs[0])
            if self.task == "classification":
                # P3 is internal: P4/P5 and the classification head all depend on it.
                return self.head(first[-1].mean(dim=(2, 3)))
            second = self.backbone(inputs[1])
            return self.decoder(tuple((a - b).abs() for a, b in zip(first, second)), inputs[0].shape[-2:])

    base.seed_everything(seed)
    return CrossModel()


def tensor_hashes(model):
    return {name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, value in model.state_dict().items()}


def control_audit(base, model, variant):
    modules = [(name, module) for name, module in model.named_modules()
               if isinstance(module, base.DS_VSSBlockSweep)]
    if variant == "original":
        if modules:
            raise RuntimeError("Original model unexpectedly contains a controlled replacement")
        return {"variant": variant, "controlled_blocks": 0}
    if len(modules) != 1 or modules[0][0] != "backbone.p3":
        raise RuntimeError(f"Expected exactly one live P3 replacement: {[name for name, _ in modules]}")
    block = modules[0][1]
    if block.control != variant:
        raise RuntimeError("Controlled variant label differs from module behavior")
    ssm = block.ds_ssm
    values = {name: float(getattr(ssm, name).detach()) for name in ("lambda_s", "alpha", "beta_i", "rho_i", "eta_i")}
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError("Nonfinite controlled scalar")
    if variant == "matched_scale075":
        for name in ("lambda_s", "alpha"):
            if name in ssm._parameters or name not in ssm._buffers or values[name] != 0.0:
                raise RuntimeError(f"Matched075 disabled constant became trainable/nonzero: {name}")
        if ssm.use_pfim or ssm.min_retain != 0.5:
            raise RuntimeError("Matched075 guidance/scale behavior changed")
    return {"variant": variant, "controlled_blocks": 1, "placement": "internal P3/8, 64 channels",
            "scalars": values, "buffer_scalars": [name for name in values if name in ssm._buffers],
            "min_retain": ssm.min_retain, "use_pfim": ssm.use_pfim}


def gradient_audit(model):
    missing, nonfinite, norms = [], [], {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
        elif not bool(parameter.grad.isfinite().all()):
            nonfinite.append(name)
        elif name.startswith(("backbone.p3.", "backbone.p4.", "backbone.p5.", "head.", "decoder.")):
            root = ".".join(name.split(".")[:2]) if name.startswith("backbone.") else name.split(".")[0]
            norms[root] = norms.get(root, 0.0) + float(parameter.grad.detach().square().sum())
    if missing or nonfinite:
        raise RuntimeError(f"Disconnected/nonfinite trainable gradients: missing={missing}, nonfinite={nonfinite}")
    for root in ("backbone.p3", "backbone.p4", "backbone.p5", "head" if hasattr(model, "head") else "decoder"):
        if norms.get(root, 0.0) <= 0:
            raise RuntimeError(f"Required live module has no nonzero gradient: {root}")
    return {"disconnected_trainable_parameters": missing, "nonfinite_gradients": nonfinite,
            "required_stage_gradient_l2": {name: math.sqrt(value) for name, value in norms.items()}}


def dataset_class(base):
    class PairedDataset(base.ManifestDataset):
        """Per-epoch/sample transform stream independent of architecture RNG draws."""
        def __init__(self, manifest, split, size, seed):
            super().__init__(manifest, split, size, train=split == "train")
            self.training_seed, self.epoch = seed, 0

        def __getitem__(self, index):
            state = random.getstate()
            try:
                random.seed((self.training_seed + 1) * 1000000007 + self.epoch * 1000003 + index)
                return super().__getitem__(index)
            finally:
                random.setstate(state)
    return PairedDataset


def make_loader(base, dataset, batch, workers, epoch, seed, shuffle):
    dataset.epoch = epoch
    generator = base.torch.Generator()
    generator.manual_seed((seed + 1) * 1000003 + epoch)
    return base.DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=workers,
                           pin_memory=True, persistent_workers=False, generator=generator, drop_last=False)


def evaluate(base, model, dataset, batch, workers, device, seed, classes, threshold=None):
    """CD metrics count unique native heldout pixels, never resize/overlap copies."""
    import numpy as np
    torch = base.torch
    loader = make_loader(base, dataset, batch, workers, 0, seed, False)
    if dataset.task == "classification":
        model.eval()
        confusion = torch.zeros((classes, classes), dtype=torch.int64)
        with torch.no_grad():
            for packed in loader:
                inputs, target = base.move_batch(packed, device)
                logits = model(*inputs)
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError("Nonfinite classification evaluation logits")
                confusion += base.confusion_from_predictions(logits.argmax(1), target, classes)
        return base.classification_metrics(confusion), None
    if not dataset.train and dataset.records[0]["split"] == "test" and threshold is None:
        raise RuntimeError("A validation-selected frozen threshold is mandatory for CD test")
    model.eval()
    scenes, offset = {}, 0
    with torch.no_grad():
        for packed in loader:
            inputs, _ = base.move_batch(packed, device)
            predictions = model(*inputs).sigmoid()
            if not bool(torch.isfinite(predictions).all()):
                raise RuntimeError("Nonfinite CD evaluation probabilities")
            for index in range(predictions.shape[0]):
                row = dataset.records[offset + index]
                key = (row["image1"], row["image2"], row["mask"])
                if key not in scenes:
                    with base.Image.open(row["mask"]) as image:
                        target = np.asarray(image.convert("L")) > 0
                    scenes[key] = {"target": target, "sum": np.zeros(target.shape, np.float64),
                                   "count": np.zeros(target.shape, np.int32)}
                x1, y1, x2, y2 = row["crop"]
                native = base.F.interpolate(predictions[index:index + 1], (y2 - y1, x2 - x1), mode="bilinear", align_corners=False)
                scenes[key]["sum"][y1:y2, x1:x2] += native[0, 0].cpu().numpy()
                scenes[key]["count"][y1:y2, x1:x2] += 1
            offset += predictions.shape[0]
    if offset != len(dataset.records):
        raise RuntimeError("CD evaluation failed to account for every frozen crop")
    probabilities, targets = [], []
    for scene in scenes.values():
        mask = scene["count"] > 0
        probabilities.append(scene["sum"][mask] / scene["count"][mask])
        targets.append(scene["target"][mask])
    probability, target = np.concatenate(probabilities), np.concatenate(targets)
    candidates = [float(threshold)] if threshold is not None else [index / 100 for index in range(10, 91, 5)]
    best, chosen = None, None
    for value in candidates:
        prediction = probability >= value
        tp, fp = int((prediction & target).sum()), int((prediction & ~target).sum())
        fn, tn = int((~prediction & target).sum()), int((~prediction & ~target).sum())
        metric = base.binary_metrics(tp, fp, fn, tn)
        metric.update({"tp": tp, "fp": fp, "fn": fn, "tn": tn, "evaluated_native_unique_pixels": int(target.size),
                       "evaluation_unit": "native_unique_pixel_mean_overlapping_probabilities"})
        if best is None or metric["f1"] > best["f1"]:
            best, chosen = metric, value
    return best, chosen


def save_checkpoint(torch, path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        raise RuntimeError(f"Unexpected checkpoint temporary file: {temporary}")
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def bound_plan(args):
    if args.validate_only:
        if args.seed != 0:
            raise RuntimeError("Preflight is seed0 only; never bypass seed1 barrier")
        return None, None
    if not args.plan or not args.trial_id:
        raise RuntimeError("Formal training requires --plan and --trial-id")
    plan = campaign.load_plan(args.plan)
    trial = campaign.trial_by_id(plan, args.trial_id)
    if trial["kind"] != "command" or trial["seed"] != args.seed or Path(trial["run_dir"]).resolve() != Path(args.output).resolve():
        raise RuntimeError("Trial kind/seed/output do not match the frozen global plan")
    if Path(plan["source_root"]).resolve() != Path(args.source_root).resolve():
        raise RuntimeError("Source code differs from the frozen global plan")
    command = trial["command"]
    for key in ("source_root", "manifest", "output", "dataset", "variant", "seed", "epochs", "patience", "batch", "imgsz"):
        flag = "--" + key.replace("_", "-")
        if command.count(flag) != 1 or command.index(flag) + 1 >= len(command):
            raise RuntimeError(f"Frozen command must explicitly specify exactly one {flag}")
        expected = command[command.index(flag) + 1]
        if str(getattr(args, key)) != str(expected):
            raise RuntimeError(f"Actual {flag} differs from the frozen command")
    frozen_data = plan.get("cross_datasets", {}).get(args.dataset)
    if not isinstance(frozen_data, dict):
        raise RuntimeError("Formal dataset is missing from the frozen cross_datasets")
    frozen_manifest = campaign.absolute(frozen_data["manifest"]).resolve()
    if frozen_manifest != Path(args.manifest).resolve():
        raise RuntimeError("Dataset manifest differs from the frozen global plan")
    for path, key in ((frozen_manifest, "manifest_sha256"),
                      (frozen_manifest.parent / "files.json", "files_sha256"),
                      (frozen_manifest.parent / "audit.json", "audit_sha256")):
        campaign.check_file({"path": str(path), "sha256": frozen_data[key]})
    campaign.verify_code(plan)
    if args.seed == 1:
        campaign.verify_barrier(plan)
    return plan, trial


def protocol(args, data, manifest):
    return {"schema": SCHEMA, "dataset": args.dataset, "task": manifest["task"],
            "logical_data_sha256": data["logical_manifest_sha256"], "split_counts": data["counts"],
            "annotation_policy": data["annotation_policy"],
            "backbone": "Mamba-YOLO-T full_t (32,64,128,256)",
            "replacement": "Exactly one internal 64-channel P3/8 VSSBlock replaced, state_dim=16; P4/P5 consume P3",
            "classification_head": "P5 global mean -> LayerNorm256 -> Dropout0.2 -> Linear(classes)",
            "change_head": "Shared Siamese encoder; absolute P2-P5 differences; UPerNet-lite width96",
            "initialization": "scratch; full baseline construction first; isolated replacement RNG; common tensor/hash matching",
            "epochs": args.epochs, "patience": args.patience, "imgsz": args.imgsz, "batch": args.batch,
            "eval_batch": args.eval_batch or args.batch, "optimizer": "AdamW", "lr": 0.0002,
            "weight_decay": 0.0005, "scheduler": "cosine to 1% initial lr", "module_warmup_epochs": 0,
            "precision": "FP32 (AMP/TF32 off)", "gradient_clip_l2": 5.0, "grad_accum": 1,
            "classification_label_smoothing": 0.1, "normalization": "fixed (RGB/255 - 0.5)/0.5",
            "augmentation": "existing ManifestDataset transformations, independent seeded epoch/sample RNG",
            "selection_metric": "macro_f1" if manifest["task"] == "classification" else "f1",
            "cd_threshold_grid": "0.10..0.90 inclusive by0.05, validation only, earliest tie",
            "cd_evaluation": "native_unique_pixel_mean_overlapping_probabilities",
            "test_policy": "Evaluate selected best checkpoint once; no test-driven control/epoch/threshold selection",
            "historical_results_comparable": False}


def run(args):
    plan, trial = bound_plan(args)
    data = verify_staged(args.manifest)
    manifest = campaign.read_json(args.manifest)
    if manifest["dataset"] != args.dataset or DATASETS[args.dataset] != manifest["task"]:
        raise RuntimeError("Dataset/manifest/task mismatch")
    base = load_runtime(args.source_root)
    torch = base.torch
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA server required; CPU/local fallback is forbidden")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_protocol = protocol(args, data, manifest)
    protocol_hash = campaign.digest(run_protocol)
    identity = {"protocol": run_protocol, "protocol_sha256": protocol_hash,
                "variant": args.variant, "seed": args.seed, "data": data,
                "source_root": str(Path(args.source_root).resolve()),
                "plan_sha256": campaign.digest(plan) if plan else None,
                "trial_id": args.trial_id, "validate_only": args.validate_only}
    campaign.frozen(output / "config.json", identity)
    complete = output / "completion.json"
    if complete.exists():
        result = campaign.read_json(complete)
        if result["identity_sha256"] != campaign.digest(identity):
            raise RuntimeError("Existing completion belongs to another trial")
        for artifact in result["artifacts"]:
            campaign.check_file(artifact)
        print(json.dumps(result), flush=True)
        return
    classes = len(manifest["classes"]) if manifest["task"] == "classification" else 1
    Dataset = dataset_class(base)
    datasets = {split: Dataset(manifest, split, args.imgsz, args.seed) for split in ("train", "val", "test")}
    if any(not len(value) for value in datasets.values()):
        raise RuntimeError("All three splits must be present; never tune CD threshold on test")
    if args.validate_only:
        return validate(base, args, manifest, datasets, data, device, output, classes)
    model = create_model(base, manifest["task"], classes, args.variant, args.seed)
    initial = tensor_hashes(model)
    reference = create_model(base, manifest["task"], classes, "original", args.seed)
    expected = tensor_hashes(reference)
    del reference
    common_keys = [name for name in expected if not name.startswith("backbone.p3.")]
    if any(initial.get(name) != expected[name] for name in common_keys):
        raise RuntimeError("Common backbone/task-head initialization differs across arms")
    model = model.to(device)
    initial_audit = {"tensor_sha256": initial, "common_tensor_count": len(common_keys),
                     "common_tensor_sha256": campaign.digest({key: initial[key] for key in common_keys}),
                     "controls": control_audit(base, model, args.variant),
                     "parameters": sum(p.numel() for p in model.parameters()),
                     "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                     "buffers": sum(p.numel() for p in model.buffers()), "hostname": socket.gethostname(),
                     "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                     "gpu": torch.cuda.get_device_name(device)}
    campaign.frozen(output / "initialization.json", initial_audit)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.0005)
    buffer_ids = {id(value) for value in model.buffers()}
    if any(id(value) in buffer_ids for group in optimizer.param_groups for value in group["params"]):
        raise RuntimeError("Inactive control buffer entered optimizer")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.000002)
    criterion = base.build_loss(manifest["task"], manifest, output, device, label_smoothing=0.1)
    last, best = output / "last.pt", output / "best.pt"
    start, best_score, best_threshold, stale, selected_epoch = 0, -math.inf, None, 0, None
    if last.exists():
        if not args.resume:
            raise RuntimeError("Partial checkpoint exists; explicit --resume required")
        checkpoint = torch.load(last, map_location=device, weights_only=False)
        if checkpoint["identity_sha256"] != campaign.digest(identity):
            raise RuntimeError("Resume checkpoint has different immutable trial identity")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start, best_score, best_threshold = checkpoint["epoch"] + 1, checkpoint["best_score"], checkpoint["threshold"]
        stale, selected_epoch = checkpoint["stale"], checkpoint["selected_epoch"]
        control_audit(base, model, args.variant)
    elif (output / "metrics.jsonl").exists():
        raise RuntimeError("Metrics exist without a resumable checkpoint; preserve evidence and use a fresh trial directory")
    for epoch in range(start, args.epochs):
        if stale >= args.patience:
            break
        base.seed_everything((args.seed + 1) * 1000003 + epoch)
        model.train()
        loader = make_loader(base, datasets["train"], args.batch, args.workers, epoch, args.seed, True)
        total_loss, samples, first_grad = 0.0, 0, None
        for index, packed in enumerate(loader):
            inputs, targets = base.move_batch(packed, device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(*inputs), targets)
            base.require_finite_loss(loss, args.dataset, args.variant, epoch + 1, index + 1)
            loss.backward()
            if index == 0 and epoch == start:
                first_grad = gradient_audit(model)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            base.require_finite_grad_norm(grad_norm, args.dataset, args.variant, epoch + 1, index + 1)
            optimizer.step()
            total_loss += float(loss.detach()) * int(targets.shape[0])
            samples += int(targets.shape[0])
        scheduler.step()
        metric, threshold = evaluate(base, model, datasets["val"], args.eval_batch or args.batch,
                                     args.workers, device, args.seed, classes)
        score = float(metric[run_protocol["selection_metric"]])
        if not math.isfinite(score):
            raise RuntimeError("Validation selection metric is nonfinite")
        improved = score > best_score
        if improved:
            best_score, best_threshold, stale, selected_epoch = score, threshold, 0, epoch + 1
        else:
            stale += 1
        audit = control_audit(base, model, args.variant)
        state = {"identity_sha256": campaign.digest(identity), "epoch": epoch,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "best_score": best_score, "threshold": best_threshold, "stale": stale,
                 "selected_epoch": selected_epoch,
                 "rng_policy": "epoch-reset architecture-independent torch/loader and epoch-sample transforms"}
        if improved:
            save_checkpoint(torch, best, state)
        save_checkpoint(torch, last, state)
        row = {"epoch": epoch + 1, "loss": total_loss / samples, "val": metric,
               "val_threshold": threshold, "best_score": best_score, "best_epoch": selected_epoch,
               "lr": scheduler.get_last_lr()[0], "controls": audit, "gradients": first_grad}
        append_json(output / "metrics.jsonl", row)
        campaign.atomic_json(output / "status.json", {"status": "training", "epoch": epoch + 1,
                              "epochs_requested": args.epochs, "seed": args.seed, "variant": args.variant,
                              "best_epoch": selected_epoch, "stale": stale, "updated_utc": campaign.utc()})
        print(f"{args.dataset} {args.variant} seed{args.seed} epoch={epoch + 1}/{args.epochs} loss={row['loss']:.6f} val={score:.6f}", flush=True)
    if not best.is_file():
        raise RuntimeError("No validation-selected checkpoint exists")
    checkpoint = torch.load(best, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    controls = control_audit(base, model, args.variant)
    best_hash = campaign.sha(best)
    test_path = output / "test_evaluation.json"
    if test_path.exists():
        tested = campaign.read_json(test_path)
        if tested["best_sha256"] != best_hash or tested["identity_sha256"] != campaign.digest(identity):
            raise RuntimeError("Stored test evaluation differs from frozen selected checkpoint")
    else:
        metric, threshold = evaluate(base, model, datasets["test"], args.eval_batch or args.batch,
                                     args.workers, device, args.seed, classes, checkpoint["threshold"])
        tested = {"identity_sha256": campaign.digest(identity), "best_sha256": best_hash,
                  "checkpoint_epoch": checkpoint["epoch"] + 1, "threshold": threshold, "metric": metric}
        campaign.atomic_json(test_path, tested)
    result = {"schema": SCHEMA, "dataset": args.dataset, "task": manifest["task"], "variant": args.variant,
              "seed": args.seed, "protocol": run_protocol, "protocol_sha256": protocol_hash,
              "data": data, "initialization_common_tensor_sha256": initial_audit["common_tensor_sha256"],
              "checkpoint": str(best), "best_sha256": best_hash, "best_epoch": tested["checkpoint_epoch"],
              "threshold": tested["threshold"], "metric": tested["metric"], "reloaded_controls": controls,
              "parameters": initial_audit["parameters"], "trainable_parameters": initial_audit["trainable_parameters"],
              "buffers": initial_audit["buffers"],
              "efficiency_status": "Params only here; matched-hardware latency and scan-aware FLOPs require the separate efficiency lane. No undercounted THOP GFLOPs claimed.",
              "historical_negative_results_preserved": True, "finished_utc": campaign.utc()}
    campaign.atomic_json(output / "result.json", result)
    artifacts = [output / name for name in ("config.json", "initialization.json", "metrics.jsonl", "test_evaluation.json", "result.json", "best.pt")]
    completion = {"schema": SCHEMA, "status": "complete", "identity_sha256": campaign.digest(identity),
                  "protocol_sha256": protocol_hash, "variant": args.variant, "seed": args.seed,
                  "artifacts": [{"path": str(path), "sha256": campaign.sha(path)} for path in artifacts],
                  "finished_utc": campaign.utc()}
    campaign.atomic_json(complete, completion)
    campaign.atomic_json(output / "status.json", {"status": "complete", "seed": args.seed, "variant": args.variant,
                          "completion_sha256": campaign.sha(complete), "finished_utc": campaign.utc()})
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


def validate(base, args, manifest, datasets, data, device, output, classes):
    """Explicit server-only preflight; never emit a formal completion receipt."""
    torch = base.torch
    rows, common_hashes, block_hashes = [], {}, {}
    for variant in VARIANTS:
        model = create_model(base, manifest["task"], classes, variant, 0)
        hashes = tensor_hashes(model)
        common_hashes[variant] = campaign.digest({name: value for name, value in hashes.items() if not name.startswith("backbone.p3.")})
        block_hashes[variant] = hashes
        model = model.to(device).train()
        loader = make_loader(base, datasets["train"], min(args.batch, 2), 0, 0, 0, False)
        inputs, target = base.move_batch(next(iter(loader)), device)
        criterion = base.build_loss(manifest["task"], manifest, output, device, label_smoothing=0.1)
        prediction = model(*inputs)
        loss = criterion(prediction, target)
        base.require_finite_loss(loss, args.dataset, variant, 0, 0)
        loss.backward()
        gradients = gradient_audit(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.0005)
        optimizer.step()
        controls = control_audit(base, model, variant)
        check = output / f"preflight_{variant}.pt"
        save_checkpoint(torch, check, {"model": model.state_dict()})
        reload_model = create_model(base, manifest["task"], classes, variant, 0).to(device)
        reload_model.load_state_dict(torch.load(check, map_location=device, weights_only=False)["model"])
        control_audit(base, reload_model, variant)
        rows.append({"variant": variant, "loss": float(loss.detach()), "logits_shape": list(prediction.shape),
                     "gradients": gradients, "controls": controls, "checkpoint_reload": True})
        del reload_model, optimizer, prediction, loss, model
        torch.cuda.empty_cache()
    if len(set(common_hashes.values())) != 1:
        raise RuntimeError("Common tensors differ across the three preflight arms")
    # Matched and full share all random initial tensors; scalar off-switches
    # lambda/alpha are deliberately zero only in Matched075.
    changed = [name for name, value in block_hashes["full"].items()
               if block_hashes["matched_scale075"].get(name) != value]
    if set(changed) != {"backbone.p3.ds_ssm.lambda_s", "backbone.p3.ds_ssm.alpha"}:
        raise RuntimeError(f"Unexpected Matched075/full initial differences: {changed}")
    campaign.atomic_json(output / "validation.json", {"schema": SCHEMA, "validation_only": True,
                          "dataset": args.dataset, "data": data, "common_tensor_sha256": common_hashes,
                          "expected_control_tensor_differences": changed, "checks": rows,
                          "status": "passed", "finished_utc": campaign.utc()})
    print(json.dumps({"status": "preflight_passed", "validation": str(output / "validation.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source-root", "manifest", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--eval-batch", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--plan")
    parser.add_argument("--trial-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch, args.imgsz) <= 0 or args.workers < 0:
        parser.error("Training dimensions/budgets must be positive and workers nonnegative")
    run(args)


if __name__ == "__main__":
    main()
