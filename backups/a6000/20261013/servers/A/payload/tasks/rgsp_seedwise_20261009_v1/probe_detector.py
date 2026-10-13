#!/usr/bin/env python3
"""One synthetic CUDA batch per full detector; no optimizer, dataset or training.

Checks the exact manifest configuration builder after the expanded control audit
has passed. This is structural/gradient validation, not detection performance or
a full training smoke run. Model construction uses its existing stride probe;
the measured forward/backward probe runs only on the explicitly assigned GPU.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from cuda_uuid_compat import ensure_uuid_cuda


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def tensors(value, torch):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for part in value.values():
            yield from tensors(part, torch)
    elif isinstance(value, (tuple, list)):
        for part in value:
            yield from tensors(part, torch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest-source", type=Path, required=True)
    parser.add_argument("--expanded-validation", type=Path, required=True)
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=128)
    args = parser.parse_args()
    require(args.imgsz >= 64 and args.imgsz % 32 == 0, "Probe image size must be at least 64 and divisible by 32")
    require(not args.output.exists(), "Refusing to overwrite detector probe evidence")
    require(args.source_root.is_absolute(), "Explicit absolute isolated source root required")
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "cases": [],
              "imgsz": args.imgsz, "batch": 1, "precision": "fp32", "optimizer_steps": 0,
              "dataset_access": False, "formal_training_started": False,
              "gradient_objective": "sum of mean squared raw detection outputs, synthetic structural check only",
              "initialization_policy": "same seed scratch; BN reset; no cross-structure common-tensor alignment",
              "probe_sha256": sha(Path(__file__)), "manifest_sha256": sha(args.manifest_source)}
    try:
        import torch
        report["cuda_identity"] = ensure_uuid_cuda(args.expected_uuid)
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        sys.path.insert(0, str(args.source_root))
        sys.path.insert(0, str(args.source_root / "tools"))
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.nn.modules.mamba_yolo import DS_VSSBlockSweep, VSSBlock
        import ultralytics.nn.modules.mamba_yolo as m
        from validate_rgsp_expanded import audit_controls

        report["source_sha256"] = sha(Path(m.__file__))
        earlier = json.loads(args.expanded_validation.read_text(encoding="utf-8"))
        require(earlier.get("status") == "PASS" and len(earlier.get("controls", [])) == 22,
                "Expanded 22-case validation has not passed")
        require(earlier["source_sha256"] == report["source_sha256"], "Module changed after expanded validation")
        report["expanded_validation_sha256"] = sha(args.expanded_validation)
        manifest_spec = importlib.util.spec_from_file_location("rgsp_probe_manifest", args.manifest_source)
        require(manifest_spec is not None and manifest_spec.loader is not None, "Cannot load manifest builder")
        manifest = importlib.util.module_from_spec(manifest_spec)
        manifest_spec.loader.exec_module(manifest)
        cases = [
            ("original", {"control": "original", "placement": "baseline", "state_dim": 0}),
            ("p2_full_d16", {"control": "full", "placement": "p2", "state_dim": 16}),
            ("p3_full_d16", {"control": "full", "placement": "p3", "state_dim": 16}),
            ("p4_full_d16", {"control": "full", "placement": "p4", "state_dim": 16}),
            ("p3_full_d8", {"control": "full", "placement": "p3", "state_dim": 8}),
            ("p3_full_d32", {"control": "full", "placement": "p3", "state_dim": 32}),
        ]
        expected_by_placement = {"p2": ("model.1", 32, 4), "p3": ("model.3", 64, 8),
                                 "p4": ("model.6", 128, 16)}
        for case, spec in cases:
            config = manifest.config_payload(spec)
            torch.manual_seed(0)
            model = DetectionModel(copy.deepcopy(config), nc=1, verbose=False)
            bn_count = 0
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.reset_running_stats()
                    require(int(torch.count_nonzero(module.running_mean)) == 0, case + ": BN running mean")
                    require(bool(torch.equal(module.running_var, torch.ones_like(module.running_var))), case + ": BN running var")
                    require(int(module.num_batches_tracked) == 0, case + ": BN batch counter")
                    bn_count += 1
            blocks = [(name, module) for name, module in model.named_modules() if isinstance(module, DS_VSSBlockSweep)]
            require(len(blocks) == (0 if case == "original" else 1), case + ": expected one actual RGSP block")
            if spec["placement"] == "p4":
                require(isinstance(model.model[5], torch.nn.Sequential), "P4 first two VSS blocks not sequential")
                require(len(model.model[5]) == 2 and all(isinstance(v, VSSBlock) for v in model.model[5]),
                        "P4 must retain exactly two ordinary VSS blocks")
                require(isinstance(model.model[6], DS_VSSBlockSweep), "P4 third block is not RGSP")
            else:
                require(isinstance(model.model[5], torch.nn.Sequential) and len(model.model[5]) == 3,
                        case + ": untouched P4 stage is not three VSS blocks")
            observations = []
            handles = []
            for name, block in blocks:
                expected_name, channels, stride = expected_by_placement[spec["placement"]]
                require(name == expected_name and block.dim == channels, case + ": actual block location/channels")
                require(block.ds_ssm.state_dim == spec["state_dim"], case + ": actual state dimension")

                def capture(unused_module, inputs, block_name=name, expected_channels=channels, expected_stride=stride):
                    value = inputs[0]
                    require(value.is_cuda, case + ": controlled block did not execute on CUDA")
                    require(tuple(value.shape) == (1, expected_channels, args.imgsz // expected_stride, args.imgsz // expected_stride),
                            case + ": block feature shape/stride mismatch")
                    observations.append({"name": block_name, "shape": list(value.shape),
                                         "device": str(value.device), "stride": expected_stride})

                handles.append(block.register_forward_pre_hook(capture))
            controls = audit_controls(model)
            model = model.to("cuda:0").train()
            model.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(0)
            image = torch.randn(1, 3, args.imgsz, args.imgsz, device="cuda:0", requires_grad=True)
            outputs = list(tensors(model(image), torch))
            require(len(outputs) == 3, case + ": expected three training-scale detection outputs")
            require(all(v.is_cuda and bool(torch.isfinite(v).all()) for v in outputs), case + ": nonfinite/non-CUDA output")
            loss = sum(v.float().square().mean() for v in outputs)
            require(bool(torch.isfinite(loss)), case + ": nonfinite loss")
            loss.backward()
            require(image.grad is not None and bool(torch.isfinite(image.grad).all()), case + ": input gradient")
            gradient_names = [name for name, value in model.named_parameters() if value.grad is not None]
            require(gradient_names, case + ": no model gradients")
            require(all(bool(torch.isfinite(value.grad).all()) for value in model.parameters() if value.grad is not None),
                    case + ": nonfinite parameter gradient")
            require(len(observations) == len(blocks), case + ": controlled block was unused/repeated")
            require([int(v) for v in model.stride.tolist()] == [8, 16, 32], case + ": detector output strides")
            for handle in handles:
                handle.remove()
            report["cases"].append({
                "case": case, "spec": spec, "status": "PASS", "actual_rgsp_blocks": len(blocks),
                "controls": controls, "forward_observations": observations,
                "output_shapes": [list(value.shape) for value in outputs],
                "detector_strides": [int(value) for value in model.stride.tolist()],
                "bn_modules_reset_and_checked": bn_count, "loss": float(loss.detach()),
                "parameters_with_gradients": len(gradient_names),
                "parameter_gradient_max_abs": max(float(value.grad.detach().abs().max()) for value in model.parameters() if value.grad is not None),
                "peak_allocated_mib": torch.cuda.max_memory_allocated(0) / 1024**2,
            })
            print(f"PASS {case}: blocks={len(blocks)} shape={[list(v.shape) for v in outputs]}", flush=True)
            del model, image, outputs, loss, blocks
            gc.collect()
            torch.cuda.empty_cache()
        report["status"] = "PASS"
    except BaseException as error:
        report["status"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        print(json.dumps({"detector_probe": report.get("status", "FAIL"), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
