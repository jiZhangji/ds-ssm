#!/usr/bin/env python3
"""Single controlled trial, with runtime checks after trainer unfreezing."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from ultralytics import YOLO
from ultralytics.nn.modules.mamba_yolo import DS_VSSBlockSweep


def reset_bn_statistics(model):
    """Discard architecture-dependent stride-probe statistics before scratch training."""
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()


def controls(model):
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, DS_VSSBlockSweep) and hasattr(module, "control"):
            s = module.ds_ssm
            expected = {
                "lambda_s": 0.0 if module.control in {"matched_identity", "matched_scale075", "energy_only", "guidance_only"} else None,
                "alpha": 0.0 if module.control in {"matched_identity", "matched_scale075", "residual_scale075", "guidance_only"} else None,
            }
            for key, value in expected.items():
                if value is not None:
                    assert key not in s._parameters, (module.control, key, "must be buffer")
                    assert float(getattr(s, key)) == value, (module.control, key, "constant changed")
            values = {key: float(getattr(s, key).detach()) for key in ("lambda_s", "alpha", "beta_i", "rho_i", "eta_i")}
            assert all(math.isfinite(v) for v in values.values()), values
            rows.append({"name": name, "control": module.control, **values,
                         "buffer_scalars": [k for k in values if k in s._buffers],
                         "min_retain": s.min_retain, "input_path": s.pfim_on_gamma and s.pfim_on_write,
                         "delta_path": s.pfim_on_delta, "fixed_importance": s.fixed_importance})
    return rows


def tensor_hashes(model):
    # State by name permits paired verification of common tensors, not just RNG seed.
    return {k: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for k, v in model.state_dict().items()}


def main():
    p = argparse.ArgumentParser()
    for key in ("config", "data", "project", "variant"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if min(args.epochs, args.patience, args.batch) <= 0:
        p.error("epochs, patience and batch must be positive")
    run_dir = Path(args.project) / args.variant
    last = run_dir / "weights/last.pt"
    if run_dir.exists() and any(run_dir.iterdir()) and not (args.resume and last.exists()):
        raise RuntimeError(f"Refusing to overwrite trial: {run_dir}")
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; no CPU fallback")
    model = YOLO(str(last) if args.resume and last.exists() else args.config)
    epoch_audits = []

    def before_setup(trainer):
        if not (args.resume and last.exists()):
            reset_bn_statistics(trainer.model)

    def setup_audit(trainer):
        report = {
            "controls": controls(trainer.model), "initial_state_sha256": tensor_hashes(trainer.model),
            "parameters": sum(p.numel() for p in trainer.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
            "buffers": sum(p.numel() for p in trainer.model.buffers()),
            "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0), "smoke_only": args.smoke,
            "resumed": bool(args.resume and last.exists()),
        }
        if args.variant != "original":
            assert len(report["controls"]) == 1, "Exactly one controlled P3 block required"
        parameter_ids = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
        for module in trainer.model.modules():
            for buffer in module.buffers(recurse=False):
                assert id(buffer) not in parameter_ids, "Buffer entered optimizer"
        (run_dir / ("runtime_audit_resume.json" if report["resumed"] else "runtime_audit.json")).write_text(json.dumps(report, indent=2))

    def epoch_audit(trainer):
        losses = [float(x) for x in trainer.tloss.detach().cpu().flatten()]
        if not all(math.isfinite(v) for v in losses):
            raise RuntimeError(f"Nonfinite training loss at epoch {trainer.epoch}: {losses}")
        row = {"epoch": trainer.epoch + 1, "loss": losses, "controls": controls(trainer.model)}
        epoch_audits.append(row)
        with (run_dir / "controls_by_epoch.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")

    model.add_callback("on_pretrain_routine_start", before_setup)
    model.add_callback("on_pretrain_routine_end", setup_audit)
    model.add_callback("on_train_epoch_end", epoch_audit)
    model.train(
        data=args.data, project=args.project, name=args.variant, exist_ok=True,
        pretrained=False, resume=str(last) if args.resume and last.exists() else False,
        device="0", epochs=2 if args.smoke else args.epochs, patience=args.patience,
        imgsz=640, batch=args.batch, workers=4,
        seed=args.seed, deterministic=True, amp=False, cache=False,
        optimizer="AdamW", lr0=0.0002, lrf=0.01, momentum=0.937, weight_decay=0.0005,
        nbs=64, warmup_epochs=3.0, warmup_bias_lr=0.1, warmup_momentum=0.8,
        module_warmup_epochs=0, cos_lr=False, lr_schedule="auto", multi_scale=False,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.1, mosaic=0.3, close_mosaic=0 if args.smoke else 15,
        degrees=180.0, translate=0.1, scale=0.3, flipud=0.5, fliplr=0.5,
        shear=0.0, perspective=0.0, mixup=0.0, copy_paste=0.0,
        box=7.5, cls=0.5, dfl=1.5, val=True, split="val", conf=None, iou=0.7,
        plots=False, save=True, save_period=-1,
    )
    assert (run_dir / "weights/best.pt").is_file()
    assert epoch_audits, "No completed training epochs"
    # Exercise the actual serialized checkpoint after final optimizer stripping.
    reloaded = YOLO(str(run_dir / "weights/best.pt"))
    checkpoint_controls = controls(reloaded.model)
    (run_dir / "train_complete.json").write_text(json.dumps({
        "finished_utc": datetime.now(timezone.utc).isoformat(), "last_epoch_observed": epoch_audits[-1]["epoch"],
        "smoke_only": args.smoke, "best_sha256": hashlib.sha256((run_dir / "weights/best.pt").read_bytes()).hexdigest(),
        "reloaded_controls": checkpoint_controls,
    }, indent=2))


if __name__ == "__main__":
    main()
