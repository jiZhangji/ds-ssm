#!/usr/bin/env python3
"""Four config-bound, full-detector covered-operation profiles; no training.

The dense forward path has fixed tensor shapes and no score-dependent pruning.
Weights are deliberately not loaded: this measures the exact frozen model config,
not detection accuracy or a new performance result for any trained checkpoint.
"""
import argparse
from contextlib import ExitStack
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys


def run(args):
    spec = importlib.util.spec_from_file_location("profile_counter", args.counter)
    counter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(counter)
    if sys.platform != "linux" or socket.gethostname() != counter.HOST:
        raise RuntimeError("Only the assigned A server may run this finite profile")
    campaign, output = args.campaign.resolve(), args.output.resolve()
    if output == campaign or campaign in output.parents:
        raise RuntimeError("Output must be outside the frozen campaign")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise FileExistsError("Existing profile output cannot be replaced")
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = counter.GPU
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    with ExitStack() as stack:
        for path in (campaign / "locks" / (counter.GPU + ".lock"), output.parent / (counter.GPU + ".lock")):
            lock = stack.enter_context(path.open("a+"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = counter.smi(["--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"])
        if any(row.startswith(counter.GPU) for row in active.splitlines()):
            raise RuntimeError("The assigned GPU became busy")
        sys.path.insert(0, str(campaign))
        import manifest
        plan = manifest.load_plan(campaign / "plan.json")
        manifest.verify_code(plan)
        sys.path.insert(0, plan["source_root"])
        import torch
        import yaml
        from cuda_uuid_compat import ensure_uuid_cuda
        identity = ensure_uuid_cuda(counter.GPU)
        from ultralytics.nn.tasks import DetectionModel
        from thop import profile
        from thop.profile import register_hooks
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        rows = []
        expected = {"cnn_yolov8n": 3011043, "original": 5985547, "matched_scale075": 5917907, "full": 5920441}
        for variant in ("cnn_yolov8n", "original", "matched_scale075", "full"):
            trial_id = f"sadd_{variant}_seed0"
            trial = next(row for row in plan["trials"] if row["id"] == trial_id)
            cfg_path = Path(trial["config"])
            cfg = yaml.safe_load(cfg_path.read_text())
            if cfg != manifest.config_payload(trial["spec"]):
                raise RuntimeError("Actual YAML differs from the frozen plan")
            torch.manual_seed(0)
            model = DetectionModel(cfg, ch=3, nc=1, verbose=False).eval().cuda()
            params = sum(p.numel() for p in model.parameters())
            if params != expected[variant]:
                raise RuntimeError(f"Unexpected registered parameter count for {variant}: {params}")
            controls = [{"name": name, "control": module.control} for name, module in model.named_modules()
                        if type(module).__name__ == "DS_VSSBlockSweep"]
            if variant in ("original", "cnn_yolov8n") and controls:
                raise RuntimeError("Unexpected RGSP replacement in baseline")
            if variant in ("matched_scale075", "full") and (len(controls) != 1 or controls[0]["control"] != variant):
                raise RuntimeError("Frozen replacement control mismatch")
            unknown = sorted({type(m).__module__ + "." + type(m).__name__ for m in model.modules()
                              if not list(m.children()) and type(m) not in register_hooks})
            inputs = (torch.zeros(1, 3, 640, 640, device="cuda"),)
            with torch.no_grad(), counter.FunctionalLedger(torch) as ledger:
                macs, _ = profile(model, inputs=inputs, verbose=False)
            for module in model.modules():
                module._buffers.pop("total_ops", None)
                module._buffers.pop("total_params", None)
            if variant == "cnn_yolov8n":
                if ledger.scan_calls or ledger.einsum_calls:
                    raise RuntimeError("CNN unexpectedly invoked SSM operations")
            elif not ledger.scan_calls or not ledger.einsum_calls:
                raise RuntimeError("Mamba scan/einsum capture incomplete")
            with torch.no_grad():
                values = model(*inputs)
                dense = values[0] if isinstance(values, tuple) else values
                finite = bool(torch.isfinite(dense).all())
            if not finite:
                raise RuntimeError("Nonfinite dense detector output")
            scan_flops = sum(call["basic_arithmetic_flops"] for call in ledger.scan_calls)
            einsum_flops = sum(call["flops"] for call in ledger.einsum_calls)
            row = {"trial_id": trial_id, "variant": variant, "parameters": params,
                   "config_path": str(cfg_path), "config_sha256": counter.sha(cfg_path), "config": cfg,
                   "trial_spec": trial["spec"], "plan_sha256": manifest.digest(plan),
                   "checkpoint_loaded": False, "initialization": "random seed0; config-bound dense forward with no value-dependent pruning",
                   "source_root": plan["source_root"], "source_inventory_verified_count": len(plan["code_hashes"]),
                   "input_shapes": [[1, 3, 640, 640]], "dense_output_shape": list(dense.shape),
                   "controls": controls, "thop_legacy_x2_flops": 2 * macs,
                   "scan_basic_arithmetic_flops": scan_flops, "functional_einsum_mac2_flops": einsum_flops,
                   "cross_merge_add_flops": ledger.cross_merge_flops,
                   "covered_flops_g": (2 * macs + scan_flops + einsum_flops + ledger.cross_merge_flops) / 1e9,
                   "scan_calls": ledger.scan_calls, "einsum_calls": ledger.einsum_calls,
                   "unregistered_leaf_module_types": unknown, "finite_forward": finite,
                   "scope": "full registered single-class 640x640 detector including dense head; excludes preprocessing/NMS"}
            counter.save(output / (variant + ".json"), row)
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("variant", "parameters", "covered_flops_g", "finite_forward")}), flush=True)
            del model, inputs, values, dense
            torch.cuda.empty_cache()
        manifest.verify_code(plan)
        contract = json.loads(args.cross_results.read_text())["counting_contract"]
        counter.save(output / "results.json", {"status": "PASS", "results": rows, "counting_contract": contract,
                        "hostname": socket.gethostname(), "pid": os.getpid(), "gpu_identity": identity,
                        "script_sha256": counter.sha(__file__), "counter_script_sha256": counter.sha(args.counter),
                        "source_inventory_before_after_verified": True, "source_inventory": plan["code_hashes"],
                        "scope_limit": "Config-bound complexity only. No new detection accuracy evaluation or checkpoint binding is claimed."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counter", type=Path, required=True)
    parser.add_argument("--cross-results", type=Path, required=True)
    run(parser.parse_args())
