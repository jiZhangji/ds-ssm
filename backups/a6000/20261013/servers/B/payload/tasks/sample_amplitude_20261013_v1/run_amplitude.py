#!/usr/bin/env python3
"""Independent SAMPLE U64 amplitude experiment; run only on an assigned Linux GPU.

This is a local controlled protocol, not an exact MLDA/ERM reproduction.  Probe
has no test-manifest argument and reads only the source training split.  A train
attempt can access its test manifest only after all 10000 optimizer updates and
the final raw checkpoint have been committed.  Existing attempts are never
resumed or overwritten; periodic last.pt is recovery evidence, not an automatic
resume interface.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import socket
import statistics
import struct
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone


SCHEMA = "sample-amplitude-u64-20261013-v1"
CLASSES = ["2s1", "bmp2", "btr70", "m1", "m2", "m35", "m548", "m60", "t72", "zsu23"]
COUNTS = {"train": [116, 55, 43, 78, 75, 76, 75, 116, 56, 116],
          "test": [58, 52, 49, 51, 53, 53, 53, 60, 52, 58]}
VARIANTS = {"Original": "original", "Matched075": "matched_scale075", "Full": "full"}
TOTAL_UPDATES, PHYSICAL_BATCH, ACCUMULATION = 10000, 16, 4
EFFECTIVE_BATCH = PHYSICAL_BATCH * ACCUMULATION
HEX64 = re.compile(r"[0-9a-f]{64}")


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def require_sha(value, label):
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ValueError("Missing/invalid SHA256: " + label)
    return value


def json_read(path):
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def json_write(path, value, replace=False):
    path = Path(path)
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if not replace:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        return
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def require_gpu_alone(gpu_uuid, initialized=False):
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                             "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True, timeout=15)
    pids = set()
    for line in result.stdout.splitlines():
        pieces = [part.strip() for part in line.split(",")]
        if len(pieces) == 2 and pieces[0] == gpu_uuid:
            pids.add(int(pieces[1]))
    if pids - {os.getpid()} or (initialized and os.getpid() not in pids):
        raise RuntimeError("Assigned GPU is not exclusively available; no takeover: " + repr(sorted(pids)))


def verify_sources(args):
    """Inventory binds actual frozen code; it must contain no data manifests."""
    inventory_path = Path(args.source_inventory).resolve()
    if sha(inventory_path) != args.source_inventory_sha256:
        raise RuntimeError("Source inventory SHA256 mismatch")
    value = json_read(inventory_path)
    rows = value if isinstance(value, list) else value.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Source inventory must be a list or an object with a files list")
    verified = {}
    for row in rows:
        path = Path(row["path"])
        if not path.is_absolute():
            raise ValueError("Inventory paths must be absolute")
        path = path.resolve()
        # This allow-list prevents a source-only probe from hashing a test/data
        # manifest supplied accidentally through the source inventory.
        if path.suffix.lower() not in (".py", ".yaml", ".yml"):
            raise ValueError("Only Python/YAML source files belong in this inventory: " + str(path))
        expected = require_sha(row["sha256"], str(path))
        if str(path) in verified:
            raise ValueError("Duplicate source inventory path: " + str(path))
        if sha(path) != expected:
            raise RuntimeError("Frozen source changed: " + str(path))
        verified[str(path)] = expected
    required = {Path(__file__).resolve()}
    required.update(Path(args.campaign_root).resolve() / name for name in
                    ("cross_trial.py", "cross_data.py", "manifest.py", "cuda_uuid_compat.py"))
    source_root = Path(args.source_root).resolve()
    if not (source_root / "tools/run_shared_sar_tasks_v2.py").is_file():
        raise FileNotFoundError("Missing frozen tools/run_shared_sar_tasks_v2.py")
    required.update(path.resolve() for path in source_root.rglob("*")
                    if path.is_file() and path.suffix.lower() in (".py", ".yaml", ".yml")
                    and "__pycache__" not in path.parts)
    missing = sorted(str(path) for path in required if str(path) not in verified)
    if missing:
        raise RuntimeError("Source inventory is incomplete: " + repr(missing))
    return [{"path": path, "sha256": verified[path]} for path in sorted(verified)]


def read_split(path, expected_sha, split):
    """Validate and cache a single split without discovering adjacent files."""
    import numpy as np
    from PIL import Image

    path = Path(path).resolve()
    if sha(path) != expected_sha:
        raise RuntimeError(split + " manifest SHA256 mismatch")
    manifest = json_read(path)
    if manifest.get("classes") != CLASSES:
        raise ValueError("Manifest class ordering differs from the frozen 10-class mapping")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != sum(COUNTS[split]):
        raise ValueError("Incorrect single-split record count: " + split)
    if manifest.get("count", len(records)) != len(records):
        raise ValueError("Manifest count field disagrees with records")
    allowed_domains = {"synth", "synthetic"} if split == "train" else {"real", "measured"}
    allowed_elevations = {14, 15, 16} if split == "train" else {17}
    seen, observed, cache, evidence = set(), [0] * len(CLASSES), [], []
    for index, row in enumerate(records):
        label = row["label"]
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(CLASSES):
            raise ValueError("Invalid integer class label")
        if row.get("class_name") != CLASSES[label] or row.get("split", split) != split:
            raise ValueError("Record class/split mismatch")
        if row.get("domain") not in allowed_domains or row.get("elevation_deg") not in allowed_elevations:
            raise ValueError("Record domain/elevation violates the frozen split")
        source_sha = require_sha(row.get("source_sha256"), "source MAT")
        if not row.get("source_mat_path"):
            raise ValueError("Missing source MAT provenance")
        image_path = Path(row["path"])
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        image_path = image_path.resolve()
        if str(image_path) in seen:
            raise ValueError("Duplicate sample path in split")
        seen.add(str(image_path))
        image_sha = require_sha(row.get("png_sha256", row.get("image_sha256", row.get("sha256"))), "PNG")
        if sha(image_path) != image_sha:
            raise RuntimeError("PNG SHA256 mismatch: " + str(image_path))
        with image_path.open("rb") as stream:
            header = stream.read(33)
        if (len(header) != 33 or header[:8] != b"\x89PNG\r\n\x1a\n" or
                header[12:16] != b"IHDR" or struct.unpack(">II", header[16:24]) != (128, 128) or
                header[24] != 16 or header[25] != 0):
            raise ValueError("Expected native 128x128 16-bit grayscale PNG: " + str(image_path))
        with Image.open(image_path) as image:
            image.load()
            array = np.asarray(image)
            mode = image.mode
        dtype_ok = array.dtype.kind == "u" and array.dtype.itemsize == 2
        pil_int32_ok = mode == "I" and array.dtype.kind == "i" and array.dtype.itemsize == 4
        if (array.shape != (128, 128) or not (dtype_ok or pil_int32_ok) or
                int(array.min()) < 0 or int(array.max()) > 65535):
            raise ValueError("PNG decoder did not preserve uint16-range integer samples")
        pixels = np.asarray(array, dtype="<u2", order="C")
        pixel_sha = hashlib.sha256(pixels.tobytes(order="C")).hexdigest()
        if pixel_sha != require_sha(row.get("pixel_sha256"), "uint16 pixel array"):
            raise RuntimeError("Canonical uint16 pixel SHA256 mismatch")
        # No RGB conversion, resizing, /255, [-1,1] transform, or target-domain fitting.
        floating = np.asarray(array, dtype=np.float32) / np.float32(65535.0)
        if floating.dtype != np.float32 or not np.isfinite(floating).all():
            raise RuntimeError("Amplitude conversion lost finite float32")
        cache.append(floating[None, ...])
        observed[label] += 1
        evidence.append({"index": index, "path": str(image_path), "label": label,
                         "png_sha256": image_sha, "pixel_sha256": pixel_sha,
                         "source_sha256": source_sha, "source_mat_path": row["source_mat_path"],
                         "decoded_mode": mode, "decoded_dtype": str(array.dtype),
                         "uint16_min": int(array.min()), "uint16_max": int(array.max()),
                         "float32_min": float(floating.min()), "float32_max": float(floating.max())})
    if observed != COUNTS[split]:
        raise ValueError("Per-class counts differ from the fixed SAMPLE selection")
    report = {"split": split, "manifest_sha256": expected_sha, "count": len(records),
              "classes": CLASSES, "class_counts": observed, "input_shape": [1, 128, 128],
              "conversion": "uint16 -> numpy.float32 / numpy.float32(65535); later repeat channel three times",
              "input_min": min(row["float32_min"] for row in evidence),
              "input_max": max(row["float32_max"] for row in evidence),
              "records": evidence, "records_sha256": digest(evidence),
              "source_mat_bytes_reopened_by_runner": False}
    return manifest, np.stack(cache), report


def verify_cached_split(path, expected_sha, report):
    if sha(path) != expected_sha:
        raise RuntimeError("Training manifest changed during the attempt")
    for row in report["records"]:
        if sha(row["path"]) != row["png_sha256"]:
            raise RuntimeError("Training input changed during the attempt: " + row["path"])


class PermutationStream:
    """A private random-permutation stream: every requested position is used."""
    def __init__(self, torch, count, seed):
        self.torch, self.count = torch, count
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed((seed + 1) * 1000003)
        self.permutation, self.position, self.permutations_started, self.consumed = [], 0, 0, 0

    def take(self, count):
        result = []
        while len(result) < count:
            if self.position == len(self.permutation):
                self.permutation = self.torch.randperm(self.count, generator=self.generator).tolist()
                self.position = 0
                self.permutations_started += 1
            stop = min(len(self.permutation), self.position + count - len(result))
            result.extend(self.permutation[self.position:stop])
            self.position = stop
        self.consumed += len(result)
        return result

    def state(self):
        return {"generator_state": self.generator.get_state(), "permutation": self.permutation,
                "position": self.position, "permutations_started": self.permutations_started,
                "consumed": self.consumed, "count": self.count}


def learning_rate(update_index):
    if not 0 <= update_index < TOTAL_UPDATES:
        raise ValueError("Optimizer update index is outside the fixed schedule")
    return 0.001 * (1.0 + math.cos(math.pi * update_index / (TOTAL_UPDATES - 1))) / 2.0


def protocol_for(args):
    # Seed, mode, executed probe budget, GPU, output and test path/hash are
    # intentionally separate attempt/evaluation bindings, not protocol changes.
    return {"schema": SCHEMA, "dataset": "SAMPLE", "method": args.method,
            "group": "U64-Clean" if args.group == "clean" else "U64-Gaussian",
            "classes": CLASSES, "train_count": 806,
            "train_manifest_sha256": args.train_manifest_sha256,
            "source_inventory_sha256": args.source_inventory_sha256,
            "input": {"native_size": [128, 128], "stored_dtype": "uint16 PNG grayscale",
                      "conversion": "numpy.float32(pixel)/numpy.float32(65535.0)",
                      "model_channels": "single amplitude channel repeated 3 times",
                      "normalization": "local per-image abs(complex_img)/max, round to uint16; all-zero -> zero",
                      "author_exact_scaling_public": False},
            "augmentation": {"geometry": "none", "noise": args.group,
                             "apply_probability": 0.5 if args.group == "gaussian" else 0.0,
                             "sigma": "per-image Uniform(0.2,0.6), standard deviation",
                             "noise_domain": "one channel, independent normal per pixel, then repeat3",
                             "clip": False, "rng": "private CPU torch generator shared across methods"},
            "training": {"initialization": "scratch", "physical_batch": PHYSICAL_BATCH,
                         "gradient_accumulation": ACCUMULATION, "samples_per_update": EFFECTIVE_BATCH,
                         "optimizer_updates": TOTAL_UPDATES, "sample_positions": 640000,
                         "sampling": "continuous private randperm stream; tail joins next permutation; drop none",
                         "training_rng": "reset Python/NumPy/CPU/CUDA to run seed after all model construction",
                         "optimizer": "SGD", "lr_initial": 0.001, "momentum": 0.9,
                         "weight_decay": 0.0005, "decay_parameters": "all trainable, including bias and norm",
                         "nesterov": False, "dampening": 0,
                         "lr_schedule": "0.001*(1+cos(pi*t/9999))/2, t=0..9999 before step",
                         "warmup_updates": 0, "module_warmup_updates": 0,
                         "loss": "cross_entropy mean; each microbatch loss / 4",
                         "label_smoothing": 0.0, "gradient_clip": None,
                         "precision": "FP32", "amp": False, "tf32": False,
                         "ema": False, "early_stopping": False,
                         "batch_norm": "physical batch16 statistics; not equivalent to physical64/36",
                         "selection": "raw model after fixed update10000; no validation set"},
            "model_source": "frozen cross_trial.load_runtime/create_model for three arms; torchvision ResNet50 v1.5",
            "literature_identity": "local controlled amplitude U64 experiment; not exact MLDA or ERM reproduction",
            "test_history": "same public benchmark already inspected in preceding diagnosis; no unseen-test claim",
            "resume_policy": "no automatic or CLI resume; preserve failed attempts and last.pt"}


def tensor_hashes(model):
    return {name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, value in model.state_dict().items()}


def construct_model(torch, cross, base, args):
    if args.method in VARIANTS:
        return cross.create_model(base, "classification", len(CLASSES), VARIANTS[args.method], args.seed)
    import torchvision
    random.seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    return torchvision.models.resnet50(weights=None, num_classes=len(CLASSES))


def imported_runtime_audit(sources, args):
    """Check live module locations, including a pre-imported wrong ultralytics."""
    expected = {row["path"]: row["sha256"] for row in sources}
    evidence = []
    for name, module in sorted(sys.modules.items()):
        controlled = (name == "ultralytics" or name.startswith("ultralytics.") or
                      name in {"rgsp_shared_v2_frozen", "sample_amplitude_frozen_cross",
                               "sample_amplitude_cuda_uuid", "cross_data", "manifest"})
        dependency = (name == "mamba_ssm" or name.startswith("mamba_ssm.") or
                      name.startswith("selective_scan") or name == "torchvision.models.resnet")
        if not (controlled or dependency) or not getattr(module, "__file__", None):
            continue
        path = Path(module.__file__).resolve()
        if path.suffix == ".pyc":
            path = Path(importlib.util.source_from_cache(str(path))).resolve()
        actual = sha(path)
        if controlled and expected.get(str(path)) != actual:
            raise RuntimeError("Imported module is outside the bound frozen source: " + name + " " + str(path))
        evidence.append({"module": name, "path": str(path), "sha256": actual,
                         "binding": "frozen source inventory" if controlled else "installed runtime dependency"})
    if args.method in VARIANTS:
        required_modules = {"ultralytics.nn.modules.mamba_yolo", "rgsp_shared_v2_frozen"}
        if not required_modules.issubset({row["module"] for row in evidence}):
            raise RuntimeError("Required frozen backbone modules were not loaded")
    return {"status": "PASS", "modules": evidence, "modules_sha256": digest(evidence),
            "installed_dependencies_are_not_author_code_frozen": True}


def gradient_check(torch, model, detailed=False):
    gradients, names, missing = [], [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                missing.append(name)
            else:
                gradients.append(parameter.grad.detach())
                names.append(name)
    if missing or not gradients:
        raise RuntimeError("Disconnected trainable gradients: " + repr(missing))
    flags = torch.stack([torch.isfinite(value).all() for value in gradients])
    if not bool(flags.all().item()):
        bad = [name for name, flag in zip(names, flags.cpu().tolist()) if not flag]
        raise FloatingPointError("Nonfinite gradients: " + repr(bad))
    result = {"all_finite": True, "missing": [], "parameter_tensors": len(gradients)}
    if detailed:
        norm_squared = float(torch.stack([value.float().square().sum() for value in gradients]).sum().item())
        if not math.isfinite(norm_squared) or norm_squared <= 0:
            raise FloatingPointError("Zero/nonfinite aggregate gradient norm")
        result["gradient_l2"] = math.sqrt(norm_squared)
    return result


def checkpoint(torch, path, model, optimizer, stream, noise_rng, args, protocol_sha, updates):
    import numpy as np
    payload = {"schema": SCHEMA, "protocol_sha256": protocol_sha, "seed": args.seed,
               "mode": args.mode, "method": args.method, "group": args.group,
               "optimizer_updates": updates, "samples_consumed": stream.consumed,
               "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
               "optimizer_state_dict": optimizer.state_dict(), "sampling_stream": stream.state(),
               "noise_rng_state": noise_rng.get_state(), "torch_rng_state": torch.get_rng_state(),
               "cuda_rng_states": torch.cuda.get_rng_state_all(), "python_rng_state": random.getstate(),
               "numpy_rng_state": np.random.get_state(), "resume_supported": False,
               "train_manifest_sha256": args.train_manifest_sha256,
               "source_inventory_sha256": args.source_inventory_sha256, "saved_utc": utc()}
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def reload_verify(torch, model, cross, base, args, path, protocol_sha, source_input):
    model.eval()
    with torch.no_grad():
        reference = model(source_input).detach().cpu()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload["protocol_sha256"] != protocol_sha or payload["seed"] != args.seed or
            payload["optimizer_updates"] != args.updates or payload["mode"] != args.mode):
        raise RuntimeError("Checkpoint binding mismatch")
    if any(value.is_floating_point() and value.dtype != torch.float32 for value in payload["state_dict"].values()):
        raise RuntimeError("Checkpoint contains a non-FP32 floating tensor")
    original_hashes = tensor_hashes(model)
    reloaded = construct_model(torch, cross, base, args).float()
    reloaded.load_state_dict(payload["state_dict"], strict=True)
    if tensor_hashes(reloaded) != original_hashes:
        raise RuntimeError("Independent checkpoint reload changed tensor bytes")
    reloaded = reloaded.to("cuda:0").eval()
    with torch.no_grad():
        actual = reloaded(source_input).detach().cpu()
    if not bool(torch.isfinite(reference).all()) or not bool(torch.isfinite(actual).all()):
        raise FloatingPointError("Nonfinite checkpoint verification output")
    agrees = bool(torch.allclose(reference, actual, rtol=1e-6, atol=1e-7))
    report = {"status": "PASS" if agrees else "FAIL", "input_split": "train",
              "input_indices": list(range(PHYSICAL_BATCH)), "independent_model_instance": True,
              "state_dict_byte_equal": True, "output_bitwise_equal": bool(torch.equal(reference, actual)),
              "output_allclose": agrees, "rtol": 1e-6, "atol": 1e-7,
              "max_abs_difference": float((reference - actual).abs().max()),
              "checkpoint_sha256": sha(path), "test_accessed": False}
    if not agrees:
        raise RuntimeError("Independent reload produced different logits: " + repr(report))
    return reloaded, report


def metrics_from_confusion(confusion):
    import numpy as np
    matrix = np.asarray(confusion, dtype=np.int64)
    support, predicted, true_positive = matrix.sum(1), matrix.sum(0), np.diag(matrix)
    precision = np.divide(true_positive, predicted, out=np.zeros(10, dtype=np.float64), where=predicted != 0)
    recall = np.divide(true_positive, support, out=np.zeros(10, dtype=np.float64), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(10, dtype=np.float64), where=(precision + recall) != 0)
    return {"accuracy": float(true_positive.sum() / matrix.sum()), "macro_precision": float(precision.mean()),
            "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean()),
            "confusion_matrix": matrix.tolist(), "support": support.tolist(),
            "predicted_counts": predicted.tolist(), "per_class_precision": precision.tolist(),
            "per_class_recall": recall.tolist(), "per_class_f1": f1.tolist()}


def verify_saved_predictions(path, manifest, expected_metrics):
    """Recompute from persisted rows with plain Python, without model inference."""
    confusion = [[0 for _ in CLASSES] for _ in CLASSES]
    count = 0
    with path.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index >= len(manifest["records"]):
                raise RuntimeError("Unexpected extra prediction row")
            row, source = json.loads(line), manifest["records"][index]
            if (row["index"] != index or row["path"] != source["path"] or
                    row["label"] != source["label"] or row["class_name"] != CLASSES[source["label"]] or
                    row["pixel_sha256"] != source["pixel_sha256"]):
                raise RuntimeError("Saved prediction does not match the frozen sample index")
            logits, probability = row["logits"], row["probabilities"]
            if (len(logits) != 10 or len(probability) != 10 or
                    not all(math.isfinite(value) for value in logits + probability) or
                    any(value < 0 or value > 1 for value in probability) or
                    abs(sum(probability) - 1.0) > 1e-5):
                raise RuntimeError("Invalid saved logits/probabilities")
            predicted = max(range(10), key=logits.__getitem__)
            if row["prediction"] != predicted or row["predicted_class"] != CLASSES[predicted]:
                raise RuntimeError("Saved class does not equal logits argmax")
            confusion[source["label"]][predicted] += 1
            count += 1
    if count != 539:
        raise RuntimeError("Incomplete saved predictions")
    support = [sum(row) for row in confusion]
    predicted_counts = [sum(confusion[truth][label] for truth in range(10)) for label in range(10)]
    precision, recall, f1 = [], [], []
    for label in range(10):
        tp = confusion[label][label]
        fp, fn = predicted_counts[label] - tp, support[label] - tp
        precision.append(tp / (tp + fp) if tp + fp else 0.0)
        recall.append(tp / (tp + fn) if tp + fn else 0.0)
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    recomputed = {"accuracy": sum(confusion[i][i] for i in range(10)) / count,
                  "macro_precision": sum(precision) / 10, "macro_recall": sum(recall) / 10,
                  "macro_f1": sum(f1) / 10, "confusion_matrix": confusion,
                  "support": support, "predicted_counts": predicted_counts,
                  "per_class_precision": precision, "per_class_recall": recall, "per_class_f1": f1}

    def numerically_equal(left, right):
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(numerically_equal(left[key], right[key]) for key in left)
        if isinstance(left, list):
            return len(left) == len(right) and all(numerically_equal(a, b) for a, b in zip(left, right))
        return math.isfinite(left) and math.isfinite(right) and abs(left - right) <= 1e-12

    if support != COUNTS["test"] or not numerically_equal(recomputed, expected_metrics):
        raise RuntimeError("Independent persisted-prediction metric recomputation disagrees")
    return {"status": "PASS", "prediction_rows": count, "prediction_sha256": sha(path),
            "method": "plain-Python row counts; per-class 2TP/(2TP+FP+FN); compared to numpy confusion metrics",
            "logits_argmax_verified": True, "metric_scale": "fraction_0_to_1",
            "recomputed_metrics": recomputed, "additional_model_evaluations": 0}


def final_test(torch, model, args, output, checkpoint_sha, protocol_sha, train_manifest, train_verified):
    if args.mode != "train" or args.updates != TOTAL_UPDATES:
        raise RuntimeError("Test access is permitted only after the full training budget")
    json_write(output / "test_access.json", {"first_access_utc": utc(), "optimizer_updates": TOTAL_UPDATES,
               "checkpoint_sha256": checkpoint_sha, "protocol_sha256": protocol_sha,
               "test_manifest_sha256": args.test_manifest_sha256, "evaluation_attempt": 1})
    # First and only test-manifest open in the entire runner occurs here.
    manifest, cached, verified = read_split(args.test_manifest, args.test_manifest_sha256, "test")
    source_paths = {row["source_mat_path"] for row in train_manifest["records"]}
    source_hashes = {row["source_sha256"] for row in train_manifest["records"]}
    image_paths = {row["path"] for row in train_verified["records"]}
    if (source_paths & {row["source_mat_path"] for row in manifest["records"]} or
            source_hashes & {row["source_sha256"] for row in manifest["records"]} or
            image_paths & {row["path"] for row in verified["records"]}):
        raise RuntimeError("Train/test path identity overlap")
    data = torch.from_numpy(cached)
    confusion = [[0] * len(CLASSES) for _ in CLASSES]
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data), PHYSICAL_BATCH):
            logits = model(data[start:start + PHYSICAL_BATCH].repeat(1, 3, 1, 1).to("cuda:0"))
            if logits.shape != (min(PHYSICAL_BATCH, len(data) - start), len(CLASSES)):
                raise RuntimeError("Unexpected classifier logits shape")
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("Nonfinite final test logits")
            probabilities = logits.softmax(1).cpu().tolist()
            logits_cpu = logits.cpu().tolist()
            predicted_labels = logits.argmax(1).cpu().tolist()
            for offset, probability in enumerate(probabilities):
                index = start + offset
                row = manifest["records"][index]
                predicted = predicted_labels[offset]
                confusion[row["label"]][predicted] += 1
                predictions.append({"index": index, "path": row["path"], "label": row["label"],
                                    "class_name": CLASSES[row["label"]], "prediction": predicted,
                                    "predicted_class": CLASSES[predicted], "probabilities": probability,
                                    "logits": logits_cpu[offset], "pixel_sha256": row["pixel_sha256"]})
    metrics = metrics_from_confusion(confusion)
    if len(predictions) != 539 or metrics["support"] != COUNTS["test"]:
        raise RuntimeError("Final evaluation did not account for all frozen test samples")
    json_write(output / "test_verified.json", verified)
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    metrics_verification = verify_saved_predictions(output / "predictions.jsonl", manifest, metrics)
    json_write(output / "metrics_verification.json", metrics_verification)
    json_write(output / "confusion_matrix.json", {"classes": CLASSES, "rows": "true", "columns": "predicted",
                                               "matrix": confusion})
    json_write(output / "test.json", {"metrics": metrics, "classes": CLASSES, "test_count": 539,
               "metric_scale": "fraction_0_to_1",
               "test_evaluated_once": True, "test_manifest_sha256": args.test_manifest_sha256,
               "checkpoint_sha256": checkpoint_sha, "protocol_sha256": protocol_sha,
               "predictions_sha256": sha(output / "predictions.jsonl"),
               "metrics_verification_sha256": sha(output / "metrics_verification.json")})
    return metrics


def ledger_verify(torch, path, updates, seed):
    """Independent read-back validates order, exposure count, LR and finite loss."""
    count, positions = 0, 0
    replay = PermutationStream(torch, 806, seed)
    with path.open("r", encoding="utf-8") as stream:
        for count, line in enumerate(stream, 1):
            row = json.loads(line)
            if (row["update"] != count or row["update_index"] != count - 1 or row["seed"] != seed or
                    row["samples"] != EFFECTIVE_BATCH or len(row["sample_indices"]) != EFFECTIVE_BATCH or
                    row["sample_indices"] != replay.take(EFFECTIVE_BATCH) or
                    row["permutations_started"] != replay.permutations_started or
                    row["permutation_position"] != replay.position or
                    row["microbatch_sizes"] != [PHYSICAL_BATCH] * ACCUMULATION or
                    row["lr"] != learning_rate(count - 1) or not math.isfinite(row["loss"]) or
                    not row["gradients_finite"]):
                raise RuntimeError("Invalid update ledger entry: " + str(count))
            positions += row["samples"]
            if row["samples_consumed"] != positions:
                raise RuntimeError("Non-contiguous cumulative sample exposure")
    if count != updates or positions != updates * EFFECTIVE_BATCH:
        raise RuntimeError("Incomplete update ledger")
    return {"status": "PASS", "updates": count, "sample_positions": positions,
            "independent_sampler_replay": True, "no_tail_sample_dropped": True,
            "first_lr": learning_rate(0), "last_lr": learning_rate(count - 1),
            "updates_sha256": sha(path)}


def run(args):
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Run models only on the assigned registered Linux server; no local fallback")
    if args.mode == "probe":
        if args.test_manifest is not None or args.test_manifest_sha256 is not None:
            raise ValueError("Source-only probe forbids test-manifest arguments")
        if not 16 <= args.updates <= 128:
            raise ValueError("Probe budget must be 16..128 updates; default is 32")
    elif args.updates != TOTAL_UPDATES or not args.test_manifest or not args.test_manifest_sha256:
        raise ValueError("Training requires exactly10000 updates and a separately bound final test manifest")
    for name in ("train_manifest_sha256", "source_inventory_sha256"):
        require_sha(getattr(args, name), name)
    if args.mode == "train":
        require_sha(args.test_manifest_sha256, "test_manifest_sha256")
    output = Path(args.output)
    if not output.is_absolute() or len(output.parts) < 5:
        raise ValueError("An explicit independent absolute output directory is required")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    args._owns_output = True
    json_write(output / "OWNER.json", {"pid": os.getpid(), "hostname": socket.gethostname(),
               "started_utc": utc(), "mode": args.mode, "seed": args.seed, "resume_supported": False})
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    require_gpu_alone(args.gpu_uuid)
    sources = verify_sources(args)
    manifest, cached, data_verified = read_split(args.train_manifest, args.train_manifest_sha256, "train")
    protocol = protocol_for(args)
    protocol_sha = digest(protocol)
    json_write(output / "protocol.json", {"protocol_sha256": protocol_sha, "protocol": protocol})
    json_write(output / "source_inventory.json", sources)
    json_write(output / "train_verified.json", data_verified)

    sys.path.insert(0, str(Path(args.campaign_root).resolve()))
    cross = load_module("sample_amplitude_frozen_cross", Path(args.campaign_root) / "cross_trial.py")
    compat = load_module("sample_amplitude_cuda_uuid", Path(args.campaign_root) / "cuda_uuid_compat.py")
    import torch
    import torchvision
    identity = compat.ensure_uuid_cuda(args.gpu_uuid)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    base = cross.load_runtime(args.source_root) if args.method in VARIANTS else None
    model = construct_model(torch, cross, base, args)
    control = cross.control_audit(base, model, VARIANTS[args.method]) if args.method in VARIANTS else {
        "implementation": "torchvision.models.resnet50", "variant": "ResNet50 v1.5", "weights": None}
    initial_hashes = tensor_hashes(model)
    common = {name: value for name, value in initial_hashes.items() if not name.startswith("backbone.p3.")}
    common_audit = {"applicable": args.method in VARIANTS}
    if args.method in VARIANTS:
        reference = cross.create_model(base, "classification", len(CLASSES), "original", args.seed)
        reference_common = {name: value for name, value in tensor_hashes(reference).items()
                            if not name.startswith("backbone.p3.")}
        if common != reference_common:
            raise RuntimeError("Common backbone/head initialization differs from the independent Original reference")
        common_audit.update({"status": "PASS", "independent_original_reference": True,
                             "common_tensors_byte_equal": True, "common_tensor_count": len(common)})
        del reference
    json_write(output / "initialization.json", {"seed": args.seed, "state_sha256": digest(initial_hashes),
               "tensors": initial_hashes, "common_tensors_excluding_p3_sha256": digest(common),
               "common_tensors_excluding_p3": common, "control": control,
               "common_initialization_audit": common_audit})
    imported = imported_runtime_audit(sources, args)
    json_write(output / "imported_runtime.json", imported)
    model = model.float().to("cuda:0")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("Non-FP32 model parameter")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=0.001, momentum=0.9, weight_decay=0.0005,
                                dampening=0, nesterov=False)
    if {id(p) for group in optimizer.param_groups for p in group["params"]} != {id(p) for p in trainable}:
        raise RuntimeError("Optimizer did not cover every trainable parameter")
    identity.update({"torchvision": torchvision.__version__, "started_utc": utc(), "pid": os.getpid(),
                     "seed": args.seed, "mode": args.mode, "executed_update_budget": args.updates,
                     "protocol_sha256": protocol_sha, "test_evaluated": False,
                     "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                     "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                     "cudnn_deterministic": torch.backends.cudnn.deterministic,
                     "bitwise_equivalence_across_devices_claimed": False,
                     "resume_supported": False})
    json_write(output / "runtime.json", identity)
    require_gpu_alone(args.gpu_uuid, initialized=True)
    data = torch.from_numpy(cached)
    targets = torch.tensor([row["label"] for row in manifest["records"]], dtype=torch.long)
    stream = PermutationStream(torch, len(data), args.seed)
    noise_rng = torch.Generator(device="cpu")
    noise_rng.manual_seed((args.seed + 1) * 2000003)
    params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in trainable)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.0)
    durations, checkpoint_durations, gradient_evidence = [], [], []
    # Frozen create_model forks the CPU RNG for its replacement block but
    # torch.manual_seed also changes CUDA state.  Reset after *all* constructors
    # so Original/Matched075/Full start Dropout from the same CUDA RNG stream.
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    training_start = time.perf_counter()
    with (output / "updates.jsonl").open("x", encoding="utf-8", buffering=1) as ledger:
        for update_index in range(args.updates):
            torch.cuda.synchronize()
            update_start = time.perf_counter()
            indices = stream.take(EFFECTIVE_BATCH)
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(update_index)
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss_total, noisy_total = 0.0, 0
            input_min, input_max = math.inf, -math.inf
            for microbatch in range(ACCUMULATION):
                chosen = indices[microbatch * PHYSICAL_BATCH:(microbatch + 1) * PHYSICAL_BATCH]
                amplitude = data[chosen].clone()
                if args.group == "gaussian":
                    applied = torch.rand(PHYSICAL_BATCH, generator=noise_rng) < 0.5
                    sigma = 0.2 + 0.4 * torch.rand(PHYSICAL_BATCH, generator=noise_rng)
                    noise = torch.randn(amplitude.shape, generator=noise_rng, dtype=torch.float32)
                    amplitude.add_(noise * sigma[:, None, None, None] * applied[:, None, None, None])
                    noisy_total += int(applied.sum())
                input_min = min(input_min, float(amplitude.min()))
                input_max = max(input_max, float(amplitude.max()))
                inputs = amplitude.repeat(1, 3, 1, 1).to("cuda:0")
                logits = model(inputs)
                if logits.shape != (PHYSICAL_BATCH, len(CLASSES)):
                    raise RuntimeError("Unexpected training logits shape")
                loss = criterion(logits, targets[chosen].to("cuda:0"))
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite training loss at update" + str(update_index + 1))
                loss_total += float(loss.detach()) / ACCUMULATION
                (loss / ACCUMULATION).backward()
            detailed = update_index == 0 or (update_index + 1) % 500 == 0 or update_index + 1 == args.updates
            gradient = gradient_check(torch, model, detailed=detailed)
            if detailed:
                if args.method in VARIANTS:
                    gradient["controlled_stages"] = cross.gradient_audit(model)
                gradient_evidence.append({"update": update_index + 1, **gradient})
            optimizer.step()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - update_start
            durations.append(seconds)
            row = {"update": update_index + 1, "update_index": update_index, "seed": args.seed,
                   "lr": lr, "loss": loss_total, "samples": EFFECTIVE_BATCH,
                   "microbatch_sizes": [PHYSICAL_BATCH] * ACCUMULATION,
                   "sample_indices": indices, "samples_consumed": stream.consumed,
                   "permutations_started": stream.permutations_started,
                   "permutation_position": stream.position, "noisy_images": noisy_total,
                   "input_min_after_augmentation": input_min, "input_max_after_augmentation": input_max,
                   "input_dtype": "float32", "model_input_channels": 3,
                   "gradients_finite": True, "update_seconds": seconds, "timestamp_utc": utc()}
            ledger.write(json.dumps(row, allow_nan=False) + "\n")
            if detailed:
                ledger.flush()
                os.fsync(ledger.fileno())
                summary = {key: row[key] for key in ("update", "loss", "lr", "samples_consumed", "update_seconds")}
                json_write(output / "progress.json", {**summary, "timestamp_utc": utc()}, replace=True)
                print(json.dumps(summary), flush=True)
            if (update_index + 1) % 500 == 0:
                checkpoint_start = time.perf_counter()
                checkpoint(torch, output / "last.pt", model, optimizer, stream, noise_rng, args,
                           protocol_sha, update_index + 1)
                checkpoint_durations.append(time.perf_counter() - checkpoint_start)
                json_write(output / "last_checkpoint.json", {"optimizer_updates": update_index + 1,
                           "path": str(output / "last.pt"), "sha256": sha(output / "last.pt"),
                           "resume_supported": False}, replace=True)
        ledger.flush()
        os.fsync(ledger.fileno())
    training_seconds = time.perf_counter() - training_start
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    ledger_audit = ledger_verify(torch, output / "updates.jsonl", args.updates, args.seed)
    json_write(output / "ledger_verification.json", ledger_audit)
    json_write(output / "gradient_audit.json", {"status": "PASS", "every_update_finite": True,
                                               "detailed_updates": gradient_evidence})
    checkpoint_path = output / ("final.pt" if args.mode == "train" else "probe.pt")
    checkpoint_start = time.perf_counter()
    if args.mode == "train":
        with (output / "last.pt").open("rb") as source, checkpoint_path.open("xb") as destination:
            shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
    else:
        checkpoint(torch, checkpoint_path, model, optimizer, stream, noise_rng, args, protocol_sha, args.updates)
        checkpoint_durations.append(time.perf_counter() - checkpoint_start)
    model, reload_report = reload_verify(torch, model, cross, base, args, checkpoint_path, protocol_sha,
                                        data[:PHYSICAL_BATCH].repeat(1, 3, 1, 1).to("cuda:0"))
    json_write(output / "reload_verification.json", reload_report)
    if args.method in VARIANTS:
        json_write(output / "final_control_audit.json", cross.control_audit(base, model, VARIANTS[args.method]))
    if verify_sources(args) != sources:
        raise RuntimeError("Frozen source changed during the attempt")
    for row in imported["modules"]:
        if sha(row["path"]) != row["sha256"]:
            raise RuntimeError("Loaded runtime module changed during the attempt: " + row["module"])
    verify_cached_split(args.train_manifest, args.train_manifest_sha256, data_verified)
    stable = durations[8:-1]
    timing = {"synchronized_cuda": True, "warmup_updates_excluded": 8,
              "final_detailed_audit_update_excluded": 1, "stable_updates": len(stable),
              "stable_update_seconds_mean": statistics.mean(stable),
              "stable_update_seconds_median": statistics.median(stable),
              "stable_update_seconds_stdev": statistics.stdev(stable) if len(stable) > 1 else 0.0,
              "training_seconds": training_seconds,
              "checkpoint_seconds_mean": statistics.mean(checkpoint_durations),
              "gpu_peak_allocated_bytes": peak_allocated, "gpu_peak_reserved_bytes": peak_reserved,
              "estimated_full_seconds": statistics.mean(stable) * TOTAL_UPDATES +
                                        statistics.mean(checkpoint_durations) * (TOTAL_UPDATES // 500),
              "estimate_scope": "10000 updates +20 checkpoint writes; excludes initial data audit, final test and reload",
              "step_scope": "CPU cached-input gather/noise + H2D +4 forward/backward +gradient finite checks +SGD; synchronized",
              "hardware": identity["physical_gpu_name"], "gpu_uuid": args.gpu_uuid}
    json_write(output / "timing.json", timing)
    metrics = {}
    if args.mode == "train":
        metrics = final_test(torch, model, args, output, reload_report["checkpoint_sha256"],
                             protocol_sha, manifest, data_verified)
    result = {"status": "PROBE_PASS" if args.mode == "probe" else "TRAINING_COMPLETE",
              "schema": SCHEMA, "dataset": "SAMPLE", "method": args.method, "group": args.group,
              "seed": args.seed, "mode": args.mode, "protocol_sha256": protocol_sha,
              "source_inventory_sha256": args.source_inventory_sha256,
              "train_manifest_sha256": args.train_manifest_sha256,
              "optimizer_updates": args.updates, "samples_consumed": stream.consumed,
              "test_evaluated": args.mode == "train", "metrics": metrics, "classes": CLASSES,
              "metric_scale": "fraction_0_to_1", "device": "cuda:0",
              "checkpoint": {"path": str(checkpoint_path), "sha256": reload_report["checkpoint_sha256"]},
              "params": params, "trainable_params": trainable_params,
              "training_seconds": training_seconds, "estimated_full_seconds": timing["estimated_full_seconds"],
              "timing": timing, "reload_verification": reload_report,
              "complexity": {"params": params, "input_shape": [1, 3, 128, 128], "flops": None,
                             "status": "FLOPS_NOT_PROFILED_SEPARATE_DIAGNOSTIC", "training_result_independent": True},
              "resume_supported": False, "finished_utc": utc()}
    if args.mode == "train":
        result["test_manifest_sha256"] = args.test_manifest_sha256
    json_write(output / "result.json", result)
    artifacts = {path.name: sha(path) for path in output.iterdir() if path.is_file() and
                 path.name not in ("OWNER.json", "completion.json") and ".tmp." not in path.name}
    json_write(output / "completion.json", {"status": result["status"], "result_sha256": sha(output / "result.json"),
               "protocol_sha256": protocol_sha, "seed": args.seed, "mode": args.mode,
               "optimizer_updates": args.updates, "test_evaluated": args.mode == "train",
               "artifact_sha256": artifacts, "finished_utc": utc()})
    print(json.dumps({"status": result["status"], "result": str(output / "result.json"),
                      "estimated_full_seconds": result["estimated_full_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("probe", "train"))
    parser.add_argument("--method", required=True, choices=(*VARIANTS, "ResNet50"))
    parser.add_argument("--group", required=True, choices=("clean", "gaussian"))
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--updates", type=int)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-inventory", required=True)
    parser.add_argument("--source-inventory-sha256", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--train-manifest-sha256", required=True)
    parser.add_argument("--test-manifest")
    parser.add_argument("--test-manifest-sha256")
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.updates is None:
        args.updates = 32 if args.mode == "probe" else TOTAL_UPDATES
    args._owns_output = False
    try:
        run(args)
    except BaseException as error:
        if args._owns_output:
            path = Path(args.output) / "failure.json"
            if not path.exists():
                json_write(path, {"status": "FAILED", "error_type": type(error).__name__, "error": str(error),
                                 "traceback": traceback.format_exc(), "finished_utc": utc(),
                                 "resume_supported": False, "existing_attempt_preserved": True})
        raise


if __name__ == "__main__":
    main()
