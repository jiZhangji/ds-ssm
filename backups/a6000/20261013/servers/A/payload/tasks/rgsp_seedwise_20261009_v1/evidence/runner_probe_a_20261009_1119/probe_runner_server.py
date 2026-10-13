#!/usr/bin/env python3
"""Independent server-only contracts and seven-class one-batch GPU probe.

No formal plan, external dataset, fit/train loop, optimizer step, download,
installation, service or process signal. Synthetic contract fixtures are not
training data and never publish to a campaign data readiness location.
"""
from __future__ import annotations

import argparse
import ast
import copy
import gc
import importlib
import importlib.util
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

EXPECTED_HELPERS = {
    "manifest.py": "ff1f6708e735d5bd91d9c341edfcfdcc714829c5ab5ee0c6110df263d5b6fed7",
    "train_trial.py": "0985a2e171d8f7a7097d5e33267c5cc980d6892f565f158c3bcfa6a0b5c5d4a3",
    "deferred_data.py": "dbfd4cadcdbe6a7533bfb679ed540d2739c8ee6b961b4558bd5e66e6e05a7fb1",
    "campaign_dispatch.py": "9def6f9d37cd8f38faf51a7812d9e3612495bc333415c46375fa28b09dc67477",
    "verify_receipts.py": "af866b84823b8d623f08bdfa6513bc987384f745ac42ac5647ffb17210d2d86d",
}


def require(value, message):
    if not value:
        raise AssertionError(message)


def rejected(label, function):
    start = time.monotonic()
    try:
        function()
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        return {"case": label, "status": "PASS", "rejected": f"{type(error).__name__}: {error}",
                "elapsed_seconds": time.monotonic() - start}
    raise AssertionError(f"Unsafe input was accepted: {label}")


def contract_checks(root, output, report, manifest):
    for filename, expected in EXPECTED_HELPERS.items():
        path = root / filename
        require(manifest.sha(path) == expected, f"Unexpected frozen helper: {path}")
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        importlib.import_module(path.stem)
    require("torch" not in sys.modules, "Control-plane imports unexpectedly initialized torch")
    report["ast_and_control_imports"] = {"status": "PASS", "torch_not_imported": True,
                                          "helper_sha256": EXPECTED_HELPERS}
    import deferred_data as deferred
    import campaign_dispatch as dispatch
    import verify_receipts as verifier

    backup = output / "pre_update" / "manifest.py"
    if backup.is_file():
        spec = importlib.util.spec_from_file_location("preserved_probe_manifest", backup)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        for control in ("original", "full"):
            arm = {"control": control, "placement": "baseline" if control == "original" else "p3"}
            require(manifest.config_payload(arm) == legacy.config_payload(arm),
                    f"Default historical configuration changed: {control}")
        report["preserved_default_configs"] = {"status": "PASS", "controls": ["original", "full"],
                                                 "old_manifest_sha256": manifest.sha(backup)}
    for control in ("original", "full", "cnn_yolov8n"):
        arm = {"control": control, "placement": "p3" if control == "full" else "baseline"}
        require(manifest.config_payload(arm)["head"][-1][3] == [1], "Default head must remain numeric one")
        require(manifest.config_payload(arm, dynamic_nc=True)["head"][-1][3] == ["nc"],
                "Deferred head must resolve the trainer-overridden nc")

    fixture = output / "synthetic_contract_fixture_not_training_data"
    descriptor_path, audit_path = fixture / "dataset.json", fixture / "source_audit.json"
    entry = {"deferred": True, "descriptor": str(descriptor_path), "audit": str(audit_path),
             "expected_identity": "SAR-ShipDataset", "expected_nc": 1, "protocol": "ssdd"}
    failures = []
    with mock.patch.object(deferred, "MAX_WAIT_SECONDS", 0):
        failures.append(rejected("missing readiness bounded without CUDA", lambda: deferred.wait_for_dataset(entry)))
    rows = []
    for split in ("train", "val", "test"):
        image_path = fixture / "images" / split / "contract_fixture.png"
        label_path = fixture / "labels" / split / "contract_fixture.txt"
        image_path.parent.mkdir(parents=True)
        label_path.parent.mkdir(parents=True)
        # Bytes exercise the inventory verifier, not an image decoder or trainer.
        with image_path.open("xb") as handle:
            handle.write(("synthetic contract fixture only: " + split).encode())
        with label_path.open("xb") as handle:
            handle.write(b"0 0.5 0.5 0.25 0.25\n")
        rows.append({"split": split, "name": image_path.name,
                     "image_sha256": manifest.sha(image_path), "label_sha256": manifest.sha(label_path)})
    data_yaml, inventory = fixture / "data.yaml", fixture / "data_content.json"
    manifest.frozen(data_yaml, {"path": str(fixture), "train": "images/train", "val": "images/val",
                               "test": "images/test", "nc": 1, "names": ["ship"]})
    manifest.frozen(inventory, rows)
    descriptor = {"root": str(fixture), "yaml": str(data_yaml), "yaml_sha256": manifest.sha(data_yaml),
                  "manifest": str(inventory), "manifest_sha256": manifest.sha(inventory),
                  "counts": {"train": 1, "val": 1, "test": 1}, "protocol": "ssdd", "nc": 1, "names": ["ship"]}
    manifest.frozen(descriptor_path, descriptor)
    manifest.frozen(audit_path, {"status": "verified", "dataset_identity": "SAR-ShipDataset", "nc": 1,
                                "names": ["ship"], "manifest_sha256": descriptor["manifest_sha256"],
                                "counts": descriptor["counts"], "synthetic_contract_fixture_only": True})
    ready = {"status": "verified_ready", "dataset_identity": "SAR-ShipDataset", "nc": 1,
             "descriptor_sha256": manifest.sha(descriptor_path), "source_audit_sha256": manifest.sha(audit_path),
             "manifest_sha256": descriptor["manifest_sha256"], "synthetic_contract_fixture_only": True}
    for key, wrong in (("dataset_identity", "SAR-AIRcraft1.0"), ("nc", 7),
                       ("descriptor_sha256", "0" * 64), ("source_audit_sha256", "1" * 64),
                       ("manifest_sha256", "2" * 64), ("status", "failed")):
        path = fixture / "rejected_cases" / (key + ".json")
        manifest.frozen(path, {**ready, key: wrong})
        changed = {**entry, "readiness": str(path)}
        failures.append(rejected("bad ready " + key, lambda changed=changed: deferred.wait_for_dataset(changed)))
    failures.append(rejected("noncontiguous class mapping", lambda: deferred.class_mapping({1: "ship"}, 1)))
    failures.append(rejected("duplicate normalized class index", lambda: deferred.class_mapping({0: "a", "0": "b"}, 2)))
    manifest.frozen(fixture / "readiness.json", ready)
    resolved, data_audit, provenance = deferred.wait_for_dataset(entry)
    require(resolved == descriptor and data_audit["counts"] == descriptor["counts"], "Valid staged fixture did not verify")
    require(provenance["class_names"] == {0: "ship"}, "Verified source class mapping changed")
    require("torch" not in sys.modules, "Deferred data checks initialized torch")
    report["deferred_contracts"] = {"status": "PASS", "synthetic_fixture_only": str(fixture),
                                    "rejections": failures, "positive_full_hash_verification": data_audit,
                                    "no_cuda_import_or_training": True}

    dead_launch = output / "control_fixture" / "dead_finalizer_launch.json"
    manifest.frozen(dead_launch, {"pid": 2147483000, "start_ticks": 1, "argv": ["absent_probe_process"]})
    dependency = {"finalizer_launch": str(dead_launch), "gpu_uuid": "GPU-contract-fixture",
                  "requires": [{"path": str(output / "control_fixture" / "missing_complete.json"),
                                "equals": {"status": "adopted_and_scheduler_retired"}}]}
    report["dead_finalizer_fail_closed"] = rejected("dead F finalizer without completion",
        lambda: dispatch.dependencies({"campaign_root": str(root)}, dependency))
    trial = {"kind": "command", "runner_kind": "deferred_detection", "run_dir": str(output / "contract_run")}
    trial["success_artifacts"] = manifest.detection_artifacts(trial)
    require(verifier.expected_artifacts(trial) == trial["success_artifacts"] and len(trial["success_artifacts"]) == 8,
            "Trainer/verifier command artifact order differs")
    report["receipt_artifact_contract"] = {"status": "PASS", "ordered_artifact_names": list(manifest.DETECTION_ARTIFACT_NAMES)}


def gpu_checks(root, source, output, gpu_uuid, report, manifest):
    from run_lane import gpu_snapshot, lock_file, wait_idle
    gpu_lock = lock_file(root / "locks" / (gpu_uuid + ".lock"))
    try:
        snapshot = wait_idle(gpu_uuid, {"idle_seconds": 15, "idle_timeout_seconds": 120},
                             output / "gpu_wait_status.json", {"scope": "independent finite nc7 probe"})
        report["gpu_before"] = snapshot
        sys.path.insert(0, str(source))
        import torch
        from cuda_uuid_compat import ensure_uuid_cuda
        report["cuda_identity"] = ensure_uuid_cuda(gpu_uuid)
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly the assigned CUDA device is required")
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        from ultralytics.models.yolo.detect import DetectionTrainer
        import ultralytics.nn.modules.mamba_yolo as modules
        sys.path.insert(0, str(source / "tools"))
        from validate_rgsp_expanded import audit_controls
        require(modules.USE_CUDA_SCAN, "Compiled selective scan must be active for this server probe")
        report["source_sha256"] = {str(path): manifest.sha(path) for path in (
            source / "ultralytics" / "nn" / "tasks.py", Path(modules.__file__),
            source / "ultralytics" / "models" / "yolo" / "detect" / "train.py",
            source / "tools" / "validate_rgsp_expanded.py", root / "cuda_uuid_compat.py", root / "run_lane.py")}
        report["torch"] = torch.__version__
        report["cuda"] = torch.version.cuda
        names = dict(enumerate(["Boeing737", "Boeing787", "A330", "A220", "A320/321", "ARJ21", "other"]))
        report["gpu_cases"] = []
        for control in ("original", "full", "cnn_yolov8n"):
            started = time.monotonic()
            torch.manual_seed(73)
            arm = {"control": control, "placement": "p3" if control == "full" else "baseline"}
            cfg = manifest.config_payload(arm, dynamic_nc=True)
            require(cfg["nc"] == 1 and cfg["head"][-1][3] == ["nc"], "Probe must exercise actual nc override")
            holder = SimpleNamespace(data={"nc": 7, "names": names}, args=SimpleNamespace(box=7.5, cls=0.5, dfl=1.5))
            model = DetectionTrainer.get_model(holder, cfg=copy.deepcopy(cfg), verbose=False)
            holder.model = model
            DetectionTrainer.set_model_attributes(holder)
            require(model.nc == model.yaml["nc"] == model.model[-1].nc == 7, "Runtime detection head did not override to seven classes")
            require(model.names == names, "Runtime class order changed")
            model = model.cuda().train()
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.reset_running_stats()
            shapes = []
            hook = model.model[-1].register_forward_hook(lambda unused_m, unused_i, outputs: shapes.extend([list(x.shape) for x in outputs]))
            image = torch.rand(2, 3, 128, 128, device="cuda", requires_grad=True)
            batch = {"img": image, "cls": torch.tensor([[0.0], [6.0]], device="cuda"),
                     "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25], [0.45, 0.55, 0.3, 0.2]], device="cuda"),
                     "batch_idx": torch.tensor([0.0, 1.0], device="cuda")}
            loss, components = model(batch)
            require(bool(torch.isfinite(loss).all()) and bool(torch.isfinite(components).all()), "Nonfinite nc7 detection loss")
            loss.backward()
            require(image.grad is not None and bool(torch.isfinite(image.grad).all()), "Nonfinite/missing input gradient")
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            require(gradients and all(bool(torch.isfinite(g).all()) for g in gradients), "Nonfinite model gradient")
            class6_grad = float(model.model[-1].cv3[0][-1].weight.grad[6].abs().sum())
            require(class6_grad > 0, "Seventh classifier output did not receive a gradient")
            controls = audit_controls(model)
            require(len(controls) == (1 if control == "full" else 0), "Wrong actual RGSP block count")
            live = gpu_snapshot(gpu_uuid)
            require({int(row[1].strip()) for row in live["compute_processes"]} == {os.getpid()},
                    "Probe CUDA process is not exclusively on the assigned UUID")
            row = {"control": control, "status": "PASS", "constructor_nc": 1, "runtime_nc": 7,
                   "actual_head_nc": model.model[-1].nc, "class_names": names, "training_feature_shapes": shapes,
                   "synthetic_image_shape": list(image.shape), "synthetic_target_class_ids": [0, 6],
                   "loss": float(loss.detach()), "loss_components": components.detach().cpu().tolist(),
                   "gradient_tensors": len(gradients), "class6_classifier_gradient_abs": class6_grad,
                   "controls": controls, "seconds": time.monotonic() - started,
                   "no_optimizer_step_no_fit_loop": True, "gpu_uuid": live["uuid"]}
            report["gpu_cases"].append(row)
            manifest.frozen(output / ("nc7_" + control + ".json"), row)
            print(json.dumps({"case": control, "status": "PASS", "seconds": row["seconds"]}), flush=True)
            hook.remove()
            del holder, model, image, batch, loss, components, gradients
            gc.collect()
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
    finally:
        gpu_lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hostname", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    require(sys.platform == "linux", "Server Linux only; never run on the workstation")
    require(socket.gethostname() == args.hostname, "Wrong explicitly selected server")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == args.gpu_uuid, "Explicit GPU UUID mask required")
    root, source, output = [Path(p).resolve() for p in (args.campaign_root, args.source_root, args.output_dir)]
    require(root in output.parents and root in source.parents, "Probe paths must stay inside the independent campaign task")
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / "probe_report.json").exists(), "Probe report already exists; never overwrite evidence")
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", YOLO_AUTOINSTALL="false", WANDB_MODE="disabled",
                      YOLO_CONFIG_DIR=str(output / "runtime-config"), MPLCONFIGDIR=str(output / "matplotlib"))
    sys.path.insert(0, str(root))
    import manifest
    report = {"schema": "rgsp-runner-probe-v1", "hostname": socket.gethostname(), "python": platform.python_version(),
              "pid": os.getpid(), "gpu_uuid": args.gpu_uuid, "started_utc": manifest.utc(),
              "probe_sha256": manifest.sha(__file__), "source_root": str(source), "output_dir": str(output),
              "scope": "synthetic control contracts and exactly three one-batch GPU forward/backward probes; no formal experiments"}
    try:
        contract_checks(root, output, report, manifest)
        gpu_checks(root, source, output, args.gpu_uuid, report, manifest)
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["finished_utc"] = manifest.utc()
        manifest.frozen(output / "probe_report.json", report)
        print(json.dumps({"status": report["status"], "report": str(output / "probe_report.json")}), flush=True)


if __name__ == "__main__":
    main()
