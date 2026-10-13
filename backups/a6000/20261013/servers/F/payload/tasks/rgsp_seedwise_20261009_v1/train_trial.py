#!/usr/bin/env python3
"""One manifest-bound detection trial with post-unfreeze audits and test evidence.

No CPU fallback, implicit resume, legacy checkpoint resume, test-driven selection,
or mutation of the original training implementation is permitted here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import sys
from pathlib import Path

from manifest import (SCHEMA, absolute, atomic_json, config_payload, detection_artifacts,
                      digest, frozen, is_detection_trial, load_plan, read_json, sha,
                      trial_by_id, utc, verify_barrier, verify_code, verify_dataset,
                      within, write_receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--resume-rich", help="Explicit compatible raw-FP32 checkpoint; legacy last.pt is refused")
    args = parser.parse_args()
    plan = load_plan(args.plan)
    trial = trial_by_id(plan, args.trial)
    if not is_detection_trial(trial):
        raise ValueError("train_trial.py accepts detection or command + runner_kind=deferred_detection")
    if trial["seed"] == 1:
        verify_barrier(plan)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must equal the explicit assigned GPU UUID")
    from run_lane import gpu_snapshot, lock_file
    snapshot = gpu_snapshot(args.gpu_uuid)
    if snapshot["compute_processes"]:
        raise RuntimeError("Assigned GPU already has a compute process; no takeover")
    verify_code(plan)
    dataset = plan["datasets"][trial["dataset"]]
    deferred_provenance = None
    if trial.get("runner_kind") == "deferred_detection":
        from deferred_data import class_mapping, wait_for_dataset
        dataset, data_audit, deferred_provenance = wait_for_dataset(dataset)
        verify_code(plan)  # The bounded data publication wait may have been long.
        if gpu_snapshot(args.gpu_uuid)["compute_processes"]:
            raise RuntimeError("Assigned GPU became occupied while waiting for data; no takeover")
    else:
        data_audit = verify_dataset(dataset)
    sys.path.insert(0, str(absolute(plan["source_root"])))
    import numpy as np
    import torch
    from cuda_uuid_compat import ensure_uuid_cuda
    cuda_identity = ensure_uuid_cuda(args.gpu_uuid)
    import yaml
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.nn.modules.mamba_yolo import DS_VSSBlockSweep
    sys.path.insert(0, str(absolute(plan["source_root"]) / "tools"))
    from validate_rgsp_expanded import audit_controls

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one designated CUDA GPU is required; no CPU fallback")
    torch.set_num_threads(4)
    with absolute(trial["config"]).open(encoding="utf-8") as handle:
        actual_config = yaml.safe_load(handle)
    if actual_config != config_payload(trial["spec"], dynamic_nc=trial.get("runner_kind") == "deferred_detection"):
        raise RuntimeError("Frozen YAML differs from the manifest's exact single-block configuration")
    with absolute(dataset["yaml"]).open(encoding="utf-8") as handle:
        data_yaml = yaml.safe_load(handle)
    if deferred_provenance is not None:
        expected_nc = deferred_provenance["nc"]
        expected_class_names = class_mapping(data_yaml.get("names"), expected_nc)
        if "nc" in data_yaml and (type(data_yaml["nc"]) is not int or data_yaml["nc"] != expected_nc):
            raise RuntimeError("Deferred dataset YAML nc differs from its audited class count")
        if deferred_provenance["class_names"] != expected_class_names:
            raise RuntimeError("Descriptor YAML class order differs from its source audit")
    yaml_root = absolute(data_yaml["path"]).resolve()
    if yaml_root != absolute(dataset["root"]).resolve():
        raise RuntimeError("The data YAML does not point to the audited staged data root")
    for split in ("train", "val", "test"):
        value = Path(data_yaml[split])
        actual = value.resolve() if value.is_absolute() else (yaml_root / value).resolve()
        if actual != (yaml_root / "images" / split).resolve():
            raise RuntimeError(f"Data YAML {split} is outside the audited split")
    run_dir = within(trial["run_dir"], plan["campaign_root"])
    trial_lock = lock_file(absolute(plan["campaign_root"]) / "locks" / (trial["id"] + ".trial.lock"))
    resuming = args.resume_rich is not None
    if run_dir.exists() and any(run_dir.iterdir()) and not resuming:
        raise RuntimeError(f"Refusing to overwrite existing trial output: {run_dir}")
    if resuming and not run_dir.is_dir():
        raise RuntimeError("Rich resume must use this trial's existing independent output directory")
    if (run_dir / "trial_complete.json").exists():
        raise RuntimeError("Completed trials may be verified/reused, not resumed")
    run_dir.mkdir(parents=True, exist_ok=True)
    smoke = bool(plan.get("smoke_only", False))
    key = dataset.get("protocol", trial["dataset"]).lower()
    if key not in {"ssdd", "sadd"}:
        raise ValueError("Detection protocol is currently frozen for SSDD and SADD only")
    epochs, patience = (150, 30) if key == "ssdd" else (200, 40)
    epochs = 2 if smoke else epochs
    spec = trial["spec"]
    expected_blocks = 0 if spec["control"] in {"original", "cnn_yolov8n"} else 1
    provenance = {"schema": SCHEMA, "plan_sha256": digest(plan), "trial_sha256": digest(trial),
                  "trial_id": trial["id"], "seed": trial["seed"], "spec": spec,
                  "config_sha256": sha(trial["config"]), "data_manifest_sha256": dataset["manifest_sha256"],
                  "data_yaml_sha256": dataset["yaml_sha256"], "code_hashes": plan["code_hashes"],
                  "epochs": epochs, "patience": patience, "smoke_only": smoke,
                  "protocol_source": "frozen tools/train_rgsp_control.py hyperparameters",
                  "fitness": "0.1*AP50+0.9*AP50_95", "selection_split": "val", "test_split": "test"}
    if deferred_provenance is not None:
        provenance["deferred_dataset"] = deferred_provenance
    frozen(run_dir / "provenance.json", provenance)

    def deferred_class_audit(model, data=None):
        if deferred_provenance is None:
            return None
        counts = {"model_nc": getattr(model, "nc", None), "head_nc": getattr(model.model[-1], "nc", None),
                  "model_yaml_nc": model.yaml.get("nc")}
        if data is not None:
            counts["data_nc"] = data.get("nc")
            if class_mapping(data.get("names"), expected_nc) != expected_class_names:
                raise RuntimeError("Runtime dataset class order differs from the descriptor YAML")
        if any(type(value) is not int or value != expected_nc for value in counts.values()):
            raise RuntimeError(f"Runtime class count differs from deferred nc={expected_nc}: {counts}")
        if class_mapping(model.names, expected_nc) != expected_class_names:
            raise RuntimeError("Runtime model class order differs from the descriptor YAML")
        return {"verified_nc": expected_nc, "class_names": expected_class_names, "observed_counts": counts}

    def tensor_hash(tensor):
        t = tensor.detach().cpu().contiguous()
        raw = t.numpy().tobytes()
        return hashlib.sha256(str(t.dtype).encode() + str(tuple(t.shape)).encode() + raw).hexdigest()

    def tensor_hashes(model):
        return {k: tensor_hash(v) for k, v in model.state_dict().items()}

    def tree_hash(tree):
        def simplify(value):
            if torch.is_tensor(value):
                return {"tensor_sha256": tensor_hash(value)}
            if isinstance(value, np.ndarray):
                return {"numpy_sha256": hashlib.sha256(value.tobytes()).hexdigest(), "dtype": str(value.dtype), "shape": list(value.shape)}
            if isinstance(value, dict):
                return {str(k): simplify(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [simplify(v) for v in value]
            if isinstance(value, np.generic):
                return value.item()
            return value
        return digest(simplify(tree))

    def cpu_tree(value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: cpu_tree(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cpu_tree(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cpu_tree(v) for v in value)
        return copy.deepcopy(value)

    def controls(model, optimizer=None):
        rows = audit_controls(model, optimizer)
        by_name = {row["name"]: row for row in rows}
        for name, module in model.named_modules():
            if not isinstance(module, DS_VSSBlockSweep):
                continue
            s = module.ds_ssm
            control = module.control
            if control != spec["control"] or int(s.state_dim) != spec.get("state_dim", 16):
                raise RuntimeError("Instantiated control/state dimension differs from plan")
            if int(s.dw.kernel_size[0]) != spec.get("trend_kernel_size", 3):
                raise RuntimeError("Trend kernel size differs from frozen spec")
            if s.pfim is not None and int(s.pfim.kernel_size) != spec.get("stats_kernel_size", 3):
                raise RuntimeError("Local statistics kernel size differs from frozen spec")
            by_name[name]["state_dim"] = int(s.state_dim)
        if len(rows) != expected_blocks:
            raise RuntimeError(f"Expected {expected_blocks} actual RGSP block, got {len(rows)}")
        return rows

    def rng_state():
        return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all()}

    def restore_rng(state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        torch.cuda.set_rng_state_all(state["torch_cuda"])

    def load_trusted_checkpoint(path):
        # Only our plan/hash-bound independent task checkpoints may reach this.
        return torch.load(path, map_location="cpu")

    def atomic_checkpoint(path, value):
        temp = path.with_name(path.name + f".tmp.{os.getpid()}")
        try:
            with temp.open("xb") as handle:
                torch.save(value, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
        finally:
            if temp.exists():
                temp.unlink()

    resume_state = None
    if resuming:
        resume_path = within(args.resume_rich, run_dir)
        if resume_path.name != "raw_fp32_last.pt":
            raise RuntimeError("Only an explicit raw_fp32_last.pt is accepted; legacy resume is never inferred")
        resume_state = load_trusted_checkpoint(resume_path)
        for k, expected in (("schema", SCHEMA), ("plan_sha256", digest(plan)), ("trial_sha256", digest(trial)),
                            ("config_sha256", sha(trial["config"])), ("data_manifest_sha256", dataset["manifest_sha256"])):
            if resume_state.get(k) != expected:
                raise RuntimeError(f"Incompatible rich checkpoint: {k}")
        if resume_state["epoch"] + 1 >= epochs or resume_state.get("stop"):
            raise RuntimeError("Checkpoint already reached terminal training; explicit finalize recovery is required")
        for k in ("model", "ema", "optimizer", "scheduler", "scaler", "rng"):
            if tree_hash(resume_state[k]) != resume_state["state_sha256"][k]:
                raise RuntimeError(f"Rich checkpoint internal restore checksum mismatch: {k}")
        old_csv = run_dir / "results.csv"
        # Preserve interrupted CSV separately; only this explicitly resumed trial is touched.
        if old_csv.exists() and old_csv.read_text(encoding="utf-8") != resume_state["results_csv"]:
            frozen(run_dir / ("interrupted_csv_" + sha(old_csv)[:16] + ".json"),
                   {"sha256": sha(old_csv), "text": old_csv.read_text(encoding="utf-8")})
        old_csv.write_text(resume_state["results_csv"], encoding="utf-8")

    class AuditedTrainer(DetectionTrainer):
        def _setup_train(self, world_size):
            super()._setup_train(world_size)
            if resume_state is None:
                return
            self.model.load_state_dict(resume_state["model"], strict=True)
            self.ema.ema.load_state_dict(resume_state["ema"], strict=True)
            self.ema.updates = resume_state["ema_updates"]
            self.optimizer.load_state_dict(resume_state["optimizer"])
            self.scheduler.load_state_dict(resume_state["scheduler"])
            self.scaler.load_state_dict(resume_state["scaler"])
            self.stopper.__dict__.update(resume_state["stopper"])
            self.best_fitness = resume_state["best_fitness"]
            self.start_epoch = resume_state["epoch"] + 1
            self.accumulate = resume_state["accumulate"]
            if self.start_epoch >= self.epochs - self.args.close_mosaic:
                self._close_dataloader_mosaic()
                self.train_loader.reset()
            generator = getattr(self.train_loader, "generator", None)
            if generator is not None and resume_state.get("loader_generator") is not None:
                generator.set_state(resume_state["loader_generator"])
            restore_rng(resume_state["rng"])
            restored = {"model": self.model.state_dict(), "ema": self.ema.ema.state_dict(),
                        "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                        "scaler": self.scaler.state_dict(), "rng": rng_state()}
            for key, value in restored.items():
                if tree_hash(value) != resume_state["state_sha256"][key]:
                    raise RuntimeError(f"State did not restore exactly: {key}")
            frozen(run_dir / (f"resume_epoch_{self.start_epoch}_audit.json"), {
                "restored_state_sha256": resume_state["state_sha256"], "checkpoint_sha256": sha(args.resume_rich),
                "controls": controls(self.model), "start_epoch": self.start_epoch,
                "limitation": "Raw FP32/EMA/optimizer/scheduler/scaler/RNG restored and checked; worker-prefetch and inherited gradient-accumulation step cursor are not bitwise replayed."})

        def save_model(self):
            super().save_model()  # preserve the frozen historical best/last/evaluation format
            state = {"schema": SCHEMA, "plan_sha256": digest(plan), "trial_sha256": digest(trial),
                     "config_sha256": sha(trial["config"]), "data_manifest_sha256": dataset["manifest_sha256"],
                     "epoch": self.epoch, "best_fitness": self.best_fitness, "stop": bool(self.stop),
                     "model": cpu_tree(self.model.state_dict()), "ema": cpu_tree(self.ema.ema.state_dict()),
                     "ema_updates": self.ema.updates, "optimizer": cpu_tree(self.optimizer.state_dict()),
                     "scheduler": cpu_tree(self.scheduler.state_dict()), "scaler": cpu_tree(self.scaler.state_dict()),
                     "stopper": copy.deepcopy(self.stopper.__dict__), "accumulate": self.accumulate,
                     "rng": cpu_tree(rng_state()), "results_csv": self.csv.read_text(encoding="utf-8"),
                     "loader_generator": cpu_tree(self.train_loader.generator.get_state()) if getattr(self.train_loader, "generator", None) is not None else None,
                     "saved_utc": utc(), "limitations": "worker prefetch and inherited last_opt_step are not replayed; not bitwise resume"}
            for name, tensor in state["model"].items():
                if tensor.is_floating_point() and tensor.dtype != torch.float32:
                    raise RuntimeError(f"Raw training checkpoint is not FP32: {name}")
            state["state_sha256"] = {k: tree_hash(state[k]) for k in ("model", "ema", "optimizer", "scheduler", "scaler", "rng")}
            atomic_checkpoint(self.wdir / "raw_fp32_last.pt", state)
            if self.best_fitness == self.fitness:
                # Keep best EMA in FP32 as additional evidence, not a different selection criterion.
                atomic_checkpoint(self.wdir / "raw_fp32_best.pt", state)
            if self.epoch == self.start_epoch:
                reloaded = load_trusted_checkpoint(self.wdir / "raw_fp32_last.pt")
                for k, expected in state["state_sha256"].items():
                    if tree_hash(reloaded[k]) != expected:
                        raise RuntimeError(f"Raw checkpoint serialization round-trip mismatch: {k}")
                frozen(run_dir / (f"raw_checkpoint_roundtrip_epoch_{self.epoch + 1}.json"),
                       {"status": "verified", "state_sha256": state["state_sha256"], "epoch": self.epoch + 1})

    model = YOLO(str(absolute(trial["config"])))
    epoch_audits = []

    def before_setup(trainer):
        for module in trainer.model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.reset_running_stats()

    def setup_audit(trainer):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
            raise RuntimeError("A backend changed the explicit GPU UUID mask")
        live_gpu = gpu_snapshot(args.gpu_uuid)
        live_pids = {int(row[1].strip()) for row in live_gpu["compute_processes"]}
        if live_pids != {os.getpid()}:
            raise RuntimeError(f"Actual CUDA process is not exclusively on its assigned UUID: {live_gpu}")
        bn_checked = 0
        for module in trainer.model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.track_running_stats:
                if (torch.count_nonzero(module.running_mean).item() != 0
                        or not torch.equal(module.running_var, torch.ones_like(module.running_var))
                        or int(module.num_batches_tracked) != 0):
                    raise RuntimeError("Stride-probe BN statistics were not fully reset before scratch training")
                bn_checked += 1
        parameter_ids = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
        for module in trainer.model.modules():
            if any(id(buffer) in parameter_ids for buffer in module.buffers(recurse=False)):
                raise RuntimeError("A frozen control buffer entered the optimizer")
        report = {"controls": controls(trainer.model, trainer.optimizer), "initial_state_sha256": tensor_hashes(trainer.model),
                  "parameters": sum(p.numel() for p in trainer.model.parameters()),
                  "trainable_parameters": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
                  "buffers": sum(p.numel() for p in trainer.model.buffers()), "torch": torch.__version__,
                  "cuda": torch.version.cuda, "python": platform.python_version(), "gpu": live_gpu,
                  "cuda_identity": cuda_identity,
                  "data_audit": data_audit, "smoke_only": smoke, "resumed": resuming,
                  "deferred_class_audit": deferred_class_audit(trainer.model, trainer.data),
                  "batchnorm_modules_verified_reset": bn_checked,
                  "initialization_policy": "seeded scratch construction; BN stride-probe statistics reset",
                  "initialization_limitations": "Same seed, not common-tensor alignment across Original/DS, placement, state dimension or differing trend-kernel shapes; initial tensor hashes are recorded.",
                  "resume_note": "on_pretrain_routine_end runs before rich-state restoration" if resuming else None}
        name = (f"runtime_audit_pre_restore_epoch_{resume_state['epoch'] + 1}_pid_{os.getpid()}.json"
                if resuming else "runtime_audit.json")
        frozen(run_dir / name, report)

    def epoch_audit(trainer):
        losses = [float(x) for x in trainer.tloss.detach().cpu().flatten()]
        if not all(math.isfinite(v) for v in losses):
            raise RuntimeError(f"Nonfinite loss at epoch {trainer.epoch + 1}")
        row = {"epoch": trainer.epoch + 1, "loss": losses, "controls": controls(trainer.model, trainer.optimizer)}
        epoch_audits.append(row)
        with (run_dir / "controls_by_epoch.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")

    model.add_callback("on_pretrain_routine_start", before_setup)
    model.add_callback("on_pretrain_routine_end", setup_audit)
    model.add_callback("on_train_epoch_end", epoch_audit)
    model.train(
        trainer=AuditedTrainer, data=dataset["yaml"], project=str(run_dir.parent), name=run_dir.name, exist_ok=True,
        pretrained=False, resume=False, device="", epochs=epochs, patience=patience,
        imgsz=640, batch=16, workers=4, seed=trial["seed"], deterministic=True, amp=False, cache=False,
        optimizer="AdamW", lr0=0.0002, lrf=0.01, momentum=0.937, weight_decay=0.0005,
        nbs=64, warmup_epochs=3.0, warmup_bias_lr=0.1, warmup_momentum=0.8,
        module_warmup_epochs=0, cos_lr=False, lr_schedule="auto", multi_scale=False,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.1, mosaic=0.3, close_mosaic=0 if smoke else 15,
        degrees=180.0, translate=0.1, scale=0.3, flipud=0.5, fliplr=0.5,
        shear=0.0, perspective=0.0, mixup=0.0, copy_paste=0.0,
        box=7.5, cls=0.5, dfl=1.5, val=True, split="val", conf=None, iou=0.7,
        plots=False, save=True, save_period=-1,
    )
    if not epoch_audits:
        raise RuntimeError("No completed training epoch was observed")
    best = run_dir / "weights" / "best.pt"
    best_hash = sha(best)
    reloaded = YOLO(str(best))
    selected_controls = controls(reloaded.model)
    selected_classes = deferred_class_audit(reloaded.model)
    frozen(run_dir / "train_complete.json", {"schema": SCHEMA, "plan_sha256": digest(plan),
           "trial_sha256": digest(trial), "finished_utc": utc(), "last_epoch_observed": epoch_audits[-1]["epoch"],
           "smoke_only": smoke, "best_sha256": best_hash, "reloaded_controls": selected_controls,
           "reloaded_class_audit": selected_classes,
           "raw_fp32_last_sha256": sha(run_dir / "weights" / "raw_fp32_last.pt"),
           "resumed": resuming})
    # This test call occurs only after the entire training/validation selection is finished.
    metrics = reloaded.val(data=dataset["yaml"], split="test", imgsz=640, batch=16, device="", workers=0,
                           half=False, plots=False, conf=0.001, iou=0.6, project=str(run_dir / "evaluation"),
                           name="test", exist_ok=True)
    overall = {"precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
               "ap50": float(metrics.box.map50), "ap75": float(metrics.box.map75), "ap50_95": float(metrics.box.map)}
    if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in overall.values()):
        raise RuntimeError(f"Nonfinite/out-of-range final metrics: {overall}")
    if sha(best) != best_hash:
        raise RuntimeError("Selected best checkpoint changed during test evaluation")
    eval_path = run_dir / "eval_test.json"
    frozen(eval_path, {"schema": SCHEMA, "plan_sha256": digest(plan), "trial_id": trial["id"],
           "trial_sha256": digest(trial), "seed": trial["seed"], "subset": trial["dataset"], "spec": spec,
           "weights": str(best), "weights_sha256": best_hash, "data": dataset["yaml"],
           "data_yaml_sha256": dataset["yaml_sha256"], "data_manifest_sha256": dataset["manifest_sha256"],
           "split": "test", "imgsz": 640, "batch": 16, "conf": 0.001, "iou": 0.6, "precision_mode": "fp32",
           "overall": overall, "speed_ms": {k: float(v) for k, v in metrics.speed.items()},
           "result_dir": str(metrics.save_dir), "smoke_only": smoke})
    frozen(run_dir / "evaluation_complete.json", {"schema": SCHEMA, "plan_sha256": digest(plan),
           "trial_id": trial["id"], "best_sha256": best_hash, "eval_test_sha256": sha(eval_path),
           "finished_utc": utc(), "status": "complete", "smoke_only": smoke})
    frozen(run_dir / "trial_complete.json", {"schema": SCHEMA, "plan_sha256": digest(plan),
           "trial_id": trial["id"], "trial_sha256": digest(trial), "status": "complete", "best_sha256": best_hash,
           "eval_test_sha256": sha(eval_path), "finished_utc": utc(), "smoke_only": smoke})
    details = {"kind": "detection", "smoke_only": smoke, "overall": overall, "resumed": resuming}
    if deferred_provenance is not None:
        details.update(kind="command", runner_kind="deferred_detection")
    write_receipt(plan, trial, detection_artifacts(trial), details)
    trial_lock.close()
    print(json.dumps({"status": "complete", "trial": trial["id"], "best_sha256": best_hash, "overall": overall}))


if __name__ == "__main__":
    main()
