#!/usr/bin/env python3
"""Run the frozen trainer, redirecting resume output/data paths only.

No architecture, optimizer, learning-rate or checkpoint tensor changes.
The legacy checkpoint contains EMA/FP16 optimizer state, not an exact RNG replay.
"""
import argparse
import ast
import json
import runpy
import sys
from pathlib import Path


def patch_resume(code, output, data, variant):
    sys.path.insert(0, str(code))
    from ultralytics.engine.trainer import BaseTrainer
    original = BaseTrainer.check_resume

    def redirected(self, overrides):
        original(self, overrides)
        if self.resume:
            self.args.project = str(output / "runs")
            self.args.name = variant
            self.args.save_dir = str(output / "runs" / variant)
            self.args.data = str(data)
            self.args.exist_ok = True

    BaseTrainer.check_resume = redirected
    return BaseTrainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--code", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--patience", type=int, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--probe", action="store_true")
    a = p.parse_args()
    sys.dont_write_bytecode = True
    data = a.output / "manifest/data.yaml"
    trainer = patch_resume(a.code, a.output, data, a.variant)
    if a.probe:
        from types import SimpleNamespace
        last = a.output / "runs" / a.variant / "weights/last.pt"
        dummy = SimpleNamespace(args=SimpleNamespace(resume=str(last)))
        trainer.check_resume(dummy, {"device": "0", "batch": 16, "imgsz": 640})
        assert dummy.resume and dummy.args.save_dir == str(a.output / "runs" / a.variant)
        assert dummy.args.data == str(data)
        assert dummy.args.epochs == a.epochs and dummy.args.seed == a.seed
        assert dummy.args.optimizer == "AdamW" and dummy.args.batch == 16
        print(json.dumps({"resume_path_probe": "PASS", "save_dir": dummy.args.save_dir,
                          "epochs": dummy.args.epochs, "seed": dummy.args.seed}))
        return
    sys.argv = [str(a.code / "tools/train_rgsp_control.py"),
                "--config", str(a.output / "manifest/configs" / (a.variant + ".yaml")),
                "--data", str(data), "--project", str(a.output / "runs"),
                "--variant", a.variant, "--seed", str(a.seed)]
    # SSDD's frozen trainer fixes 150/30/16 internally; SADD exposes these flags.
    source = ast.parse((a.code / "tools/train_rgsp_control.py").read_text())
    supports_epochs = any(isinstance(n, ast.Constant) and n.value == "--epochs"
                          for n in ast.walk(source))
    if supports_epochs:
        sys.argv.extend(["--epochs", str(a.epochs), "--patience", str(a.patience), "--batch", "16"])
    elif (a.epochs, a.patience) != (150, 30):
        raise RuntimeError("Frozen SSDD trainer requires epochs=150 and patience=30")
    if a.resume:
        sys.argv.append("--resume")
    runpy.run_path(str(a.code / "tools/train_rgsp_control.py"), run_name="__main__")


if __name__ == "__main__":
    main()
