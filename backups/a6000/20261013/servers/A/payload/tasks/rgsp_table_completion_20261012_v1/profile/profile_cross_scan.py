#!/usr/bin/env python3
"""Finite server-only THOP profile of the nine frozen seed0 cross-task models.

No training, predictions, package installation, or mutation of frozen artifacts.
Adds actual forward-call scan/einsum counts to the legacy THOP x2 convention.
The reported value is a covered-operation estimate, never a hardware instruction
count or a claim that every floating-point operation has been counted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback

GPU = "GPU-12cf4841-46b9-94d1-2715-9932134cf964"
HOST = "powerleader-PT6620W"
VARIANTS = ("original", "matched_scale075", "full")
DATASETS = (("sar_acd", "classification", 5, 224),
            ("sample", "classification", 10, 224),
            ("ottawa", "change_detection", 1, 192))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def smi(arguments):
    return subprocess.check_output(["nvidia-smi", *arguments], text=True, timeout=15).strip()


class FunctionalLedger:
    def __init__(self, torch):
        self.torch = torch
        self.scan_calls = []
        self.einsum_calls = []
        self.cross_merge_flops = 0
        self.restore = []

    def __enter__(self):
        torch = self.torch
        original_einsum = torch.einsum

        def einsum(equation, *operands):
            tensors = operands[0] if len(operands) == 1 and isinstance(operands[0], (tuple, list)) else operands
            compact = equation.replace(" ", "")
            left, right = compact.split("->")
            terms = left.split(",")
            if len(terms) != 2 or len(tensors) != 2 or "." in compact:
                raise RuntimeError(f"Unreviewed einsum: {equation}")
            dimensions = {}
            for term, tensor in zip(terms, tensors):
                if len(term) != tensor.ndim or len(set(term)) != len(term):
                    raise RuntimeError(f"Unreviewed einsum shape: {equation}")
                for name, size in zip(term, tensor.shape):
                    if name in dimensions and dimensions[name] != int(size):
                        raise RuntimeError("Einsum broadcasting is outside this counter")
                    dimensions[name] = int(size)
            # Standard dense-contraction convention: each multiply-accumulate = 2.
            flops = 2 * math.prod(dimensions.values())
            self.einsum_calls.append({"equation": equation, "shapes": [list(t.shape) for t in tensors], "flops": flops})
            return original_einsum(equation, *operands)

        torch.einsum = einsum
        self.restore.append(lambda: setattr(torch, "einsum", original_einsum))
        common = importlib.import_module("ultralytics.nn.modules.common_utils_mbyolo")
        self.wrap_scan(common.SelectiveScanCore, "core")
        try:
            interface = importlib.import_module("mamba_ssm.ops.selective_scan_interface")
        except ImportError:
            interface = None
        if interface is not None and hasattr(interface, "SelectiveScanFn"):
            self.wrap_scan(interface.SelectiveScanFn, "mamba")
        original_merge = common.CrossMerge.forward

        def merge(ctx, value):
            batch, directions, channels, height, width = map(int, value.shape)
            if directions != 4:
                raise RuntimeError("Only the frozen four-direction CrossMerge was reviewed")
            self.cross_merge_flops += 3 * batch * channels * height * width
            return original_merge(ctx, value)

        common.CrossMerge.forward = staticmethod(merge)
        self.restore.append(lambda: setattr(common.CrossMerge, "forward", staticmethod(original_merge)))
        return self

    def wrap_scan(self, cls, backend):
        original = cls.forward

        def wrapped(ctx, *args, **kwargs):
            # Both reviewed backends have u, delta, A, B, C as their first args.
            u, delta, a, b, c = args[:5]
            batch, channels, length = map(int, u.shape)
            states = int(a.shape[-1])
            d = args[5] if len(args) > 5 else kwargs.get("D")
            z = (args[6] if len(args) > 6 else kwargs.get("z")) if backend == "mamba" else None
            bias_index, softplus_index = (7, 8) if backend == "mamba" else (6, 7)
            bias = args[bias_index] if len(args) > bias_index else kwargs.get("delta_bias")
            softplus = args[softplus_index] if len(args) > softplus_index else kwargs.get("delta_softplus", False)
            if z is not None:
                raise RuntimeError("Internal z gating is outside the reviewed scan counter")
            if b.ndim not in (3, 4) or c.ndim not in (3, 4):
                raise RuntimeError("Only real variable-B/C scan is covered")
            if a.is_complex() or list(delta.shape) != list(u.shape):
                raise RuntimeError("Unexpected scan dtype/shape")
            elements = batch * channels * length
            # Algebraically equivalent factored recurrence:
            # delta*u:1; delta*A:N; B*(delta*u):N; a*h+b:2N;
            # dot(C,h):2N-1. Sum = 6N basic scalar operations per token/channel.
            # Add skip D*u (2), delta bias (1), softplus addition (1) if active.
            arithmetic = elements * (6 * states + 2 * int(d is not None) + int(bias is not None) + int(bool(softplus)))
            special = elements * (states + 2 * int(bool(softplus)))
            self.scan_calls.append({"backend": backend, "u_shape": list(u.shape), "a_shape": list(a.shape),
                                    "b_shape": list(b.shape), "c_shape": list(c.shape),
                                    "has_D": d is not None, "has_delta_bias": bias is not None,
                                    "delta_softplus": bool(softplus), "basic_arithmetic_flops": arithmetic,
                                    "special_function_evaluations_symbolic_upper_bound": special})
            return original(ctx, *args, **kwargs)

        cls.forward = staticmethod(wrapped)
        self.restore.append(lambda: setattr(cls, "forward", staticmethod(original)))

    def __exit__(self, *_):
        for restore in reversed(self.restore):
            restore()


def run(args):
    campaign = args.campaign.resolve()
    output = args.output.resolve()
    if sys.platform != "linux" or socket.gethostname() != HOST or args.expected_gpu != GPU:
        raise RuntimeError("This finite profile is pinned to the assigned A server/GPU")
    if campaign == output or campaign in output.parents:
        raise RuntimeError("Results must not modify the frozen campaign")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise FileExistsError("Use a new output directory; never replace an earlier run")
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    with ExitStack() as stack:
        for path in (campaign / "locks" / (GPU + ".lock"), output.parent / (GPU + ".lock")):
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context(path.open("a+"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = smi(["--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"])
        if any(row.startswith(GPU) for row in active.splitlines()):
            raise RuntimeError("Assigned GPU became busy; refusing to overlap another job")
        sys.path.insert(0, str(campaign))
        import torch
        from cuda_uuid_compat import ensure_uuid_cuda
        identity = ensure_uuid_cuda(GPU)
        import cross_trial
        from thop import profile
        from thop.profile import register_hooks
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        plan = json.loads((campaign / "plan.json").read_text())
        source = Path(plan["source_root"])
        source_evidence = {str(path): sha(path) for path in
                           (campaign / "cross_trial.py", source / "tools/run_shared_sar_tasks_v2.py",
                            source / "ultralytics/nn/modules/mamba_yolo.py",
                            source / "ultralytics/nn/modules/common_utils_mbyolo.py")}
        base = cross_trial.load_runtime(source)
        rows = []
        for dataset, task, classes, size in DATASETS:
            for variant in VARIANTS:
                trial_id = f"{dataset}_{variant}_seed0"
                run_dir = campaign / "runs" / trial_id
                config_path, result_path = run_dir / "config.json", run_dir / "result.json"
                config, result = (json.loads(path.read_text()) for path in (config_path, result_path))
                if config["variant"] != variant or config["protocol"]["imgsz"] != size or result["task"] != task:
                    raise RuntimeError(f"Frozen configuration mismatch: {trial_id}")
                checkpoint_path = Path(result["checkpoint"])
                checkpoint_sha = sha(checkpoint_path)
                if checkpoint_sha != result["best_sha256"]:
                    raise RuntimeError(f"Selected checkpoint hash mismatch: {trial_id}")
                model = cross_trial.create_model(base, task, classes, variant, 0)
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                model.load_state_dict(checkpoint["model"], strict=True)
                parameters = sum(p.numel() for p in model.parameters())
                if parameters != result["parameters"]:
                    raise RuntimeError(f"Parameter count differs from training result: {trial_id}")
                control = cross_trial.control_audit(base, model, variant)
                model = model.eval().cuda()
                inputs = tuple(torch.zeros(1, 3, size, size, device="cuda") for _ in range(1 if task == "classification" else 2))
                unknown_leaf_modules = sorted({type(m).__module__ + "." + type(m).__name__ for m in model.modules()
                                               if not list(m.children()) and type(m) not in register_hooks})
                with torch.no_grad(), FunctionalLedger(torch) as ledger:
                    macs, _ = profile(model, inputs=inputs, verbose=False)
                scan_flops = sum(row["basic_arithmetic_flops"] for row in ledger.scan_calls)
                einsum_flops = sum(row["flops"] for row in ledger.einsum_calls)
                if not ledger.scan_calls or not ledger.einsum_calls:
                    raise RuntimeError(f"Scan/einsum capture was incomplete: {trial_id}")
                with torch.no_grad():
                    values = model(*inputs)
                    finite = bool(torch.isfinite(values).all())
                if not finite:
                    raise RuntimeError(f"Nonfinite output: {trial_id}")
                row = {"trial_id": trial_id, "dataset": dataset, "task": task, "variant": variant,
                       "parameters": parameters, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                       "input_shapes": [list(x.shape) for x in inputs], "output_shape": list(values.shape),
                       "checkpoint_sha256": checkpoint_sha, "checkpoint_path": str(checkpoint_path),
                       "result_sha256": sha(result_path), "config_sha256": sha(config_path),
                       "control_audit": control, "thop_legacy_x2_flops": 2 * macs,
                       "scan_basic_arithmetic_flops": scan_flops, "functional_einsum_mac2_flops": einsum_flops,
                       "cross_merge_add_flops": ledger.cross_merge_flops,
                       "covered_flops_g": (2 * macs + scan_flops + einsum_flops + ledger.cross_merge_flops) / 1e9,
                       "scan_calls": ledger.scan_calls, "einsum_calls": ledger.einsum_calls,
                       "unregistered_leaf_module_types": unknown_leaf_modules,
                       "finite_forward": finite, "scope": "full registered task model; paired CD encoders counted twice, shared parameters once"}
                save(output / (trial_id + ".json"), row)
                rows.append(row)
                print(json.dumps({k: row[k] for k in ("trial_id", "parameters", "covered_flops_g", "finite_forward")}), flush=True)
                del model, checkpoint, values, inputs
                torch.cuda.empty_cache()
        if any(sha(path) != expected for path, expected in source_evidence.items()):
            raise RuntimeError("Frozen source changed during profiling")
        report = {"status": "PASS", "hostname": socket.gethostname(), "pid": os.getpid(), "gpu_identity": identity,
                  "finished_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(__file__),
                  "source_sha256": source_evidence, "torch_version": torch.__version__,
                  "thop_source_sha256": sha(inspect.getsourcefile(profile)), "results": rows,
                  "counting_contract": {"base": "2 times existing THOP counters, the project's legacy convention; normalization uses THOP approximation",
                    "einsum": "2*product(unique index dimensions), standard dense contraction MAC convention",
                    "scan": "6*B*D*L*N factored real variable-B/C recurrence arithmetic; +BDL for delta_bias; +2BDL for D*u; +BDL for softplus addition",
                    "cross_merge": "3*B*D*H*W scalar additions",
                    "not_included": ["scan exponentials/softplus special functions (symbolic counts supplied per call)",
                      "all other functional elementwise arithmetic, activations, statistics, and interpolation not covered by THOP",
                      "CUDA parallel-prefix overhead, address arithmetic, memory movement, preprocessing and task postprocessing"],
                    "interpretation": "covered-operation estimate for complete models; not exact total FLOPs, latency or GPU instruction count; do not compare as identical coverage to legacy THOP-only numbers"}}
        save(output / "results.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu", required=True)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
