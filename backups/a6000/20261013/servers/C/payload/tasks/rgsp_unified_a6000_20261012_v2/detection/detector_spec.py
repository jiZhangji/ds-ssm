"""Frozen detection structures and the new, independent 2026-10-12 protocol.

This module is standard-library-only. Importing it cannot run a model.
YOLOv8s support does not add it to the original SSDD candidate matrix.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


SCHEMA = "rgsp-unified-detection-20261012-v1"
METHODS = {
    "Original": "original",
    "Matched075": "matched_scale075",
    "Full": "full",
    "YOLOv8n": "cnn_yolov8n",
    "YOLOv8s": "cnn_yolov8s",
}
ALIASES = {key.lower(): key for key in METHODS}
ALIASES.update({value: key for key, value in METHODS.items()})
EXPECTED_COUNTS = {
    "ssdd": {"train": 730, "val": 186, "test": 232},
    "sadd": {"train": 2372, "val": 296, "test": 298},
    "sar_ship": {"train": 31735, "val": 3970, "test": 3974},
}
PROTOCOL = {
    "schema": SCHEMA,
    "new_independent_protocol": True,
    "historical_performance_reuse": False,
    "physical_batch": 16,
    "gradient_accumulation": 4,
    "nominal_effective_batch": 64,
    "tail_policy": "flush each epoch; divide native summed-loss gradients by actual accumulated sample count",
    "batch_norm_policy": "physical microbatch statistics; not equivalent to physical batch64",
    "epochs": 150,
    "patience": 30,
    "optimizer": "AdamW",
    "lr0": 0.0002,
    "betas": [0.937, 0.999],
    "epsilon": 1e-8,
    "weight_decay": {"weights": 0.0005, "bias_and_norm": 0.0},
    "optimizer_group_policy": "any local parameter name containing bias, plus norm parameters, have zero weight decay",
    "lr_schedule": "3 actual full epochs of linear warmup 0->lr0, then linear to lr0*0.01 at planned epoch150 end",
    "warmup_minimum_steps": None,
    "warmup_bias_lr_special_case": False,
    "module_warmup_epochs": 0,
    "gradient_clip_l2": 10.0,
    "initialization": "scratch; same-seed Mamba common tensors outside model.3 paired; stride-probe BN reset",
    "precision": "FP32 train/checkpoint/evaluation; AMP and TF32 off",
    "input": [3, 640, 640],
    "augmentation": {
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.1,
        "degrees": 180.0, "translate": 0.1, "scale": 0.3,
        "shear": 0.0, "perspective": 0.0,
        "flipud": 0.5, "fliplr": 0.5, "mosaic": 0.3,
        "mixup": 0.0, "copy_paste": 0.0,
        "close_mosaic_last_epochs": 15, "multi_scale": False,
    },
    "loss": {"box": 7.5, "cls": 0.5, "dfl": 1.5},
    "selection": "validation EMA fitness=0.1*AP50+0.9*AP50_95; earliest exact tie",
    "test_policy": "only selected best FP32 checkpoint; never used for training, threshold tuning or selection",
    "evaluation": {
        "confidence": 0.001, "nms_iou": 0.7, "max_det": 300,
        "batch": 16, "P_R": "maximum-F1 curve summary, not fixed deployment threshold",
        "AP": "class-mean AP across IoU 0.50:0.05:0.95; AP75 is index5",
        "unfused": True,
    },
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def method_name(value):
    try:
        return ALIASES[value.lower()]
    except KeyError as error:
        raise ValueError("Unsupported method; no surrogate fallback: " + value) from error


def update_budget(train_count):
    if train_count < 16:
        raise ValueError("Physical batch16 requires at least 16 training examples")
    microbatches = math.ceil(train_count / 16)
    updates = math.ceil(microbatches / 4)
    return {"microbatches_per_epoch": microbatches, "updates_per_epoch": updates,
            "warmup_updates": 3 * updates, "planned_total_updates": 150 * updates}


def learning_rate(update_number, budget):
    """One-based update number; exactly three real epochs, no 100-step floor."""
    warmup, total = budget["warmup_updates"], budget["planned_total_updates"]
    if not 1 <= update_number <= total:
        raise ValueError("Optimizer-update cursor outside frozen budget")
    if update_number <= warmup:
        return PROTOCOL["lr0"] * update_number / warmup
    return PROTOCOL["lr0"] * (1.0 - 0.99 * (update_number - warmup) / (total - warmup))


def model_config(method, nc):
    """Exact local YOLOv8 n/s or Mamba-YOLO-T with one real P3 replacement."""
    method = method_name(method)
    if type(nc) is not int or nc <= 0:
        raise ValueError("Positive explicit class count required")
    control = METHODS[method]
    if method.startswith("YOLOv8"):
        scale = "n" if method == "YOLOv8n" else "s"
        return {
            "nc": nc, "scales": {scale: [0.33, 0.25 if scale == "n" else 0.50, 1024]}, "scale": scale,
            "backbone": [
                [-1, 1, "Conv", [64, 3, 2]], [-1, 1, "Conv", [128, 3, 2]],
                [-1, 3, "C2f", [128, True]], [-1, 1, "Conv", [256, 3, 2]],
                [-1, 6, "C2f", [256, True]], [-1, 1, "Conv", [512, 3, 2]],
                [-1, 6, "C2f", [512, True]], [-1, 1, "Conv", [1024, 3, 2]],
                [-1, 3, "C2f", [1024, True]], [-1, 1, "SPPF", [1024, 5]],
            ],
            "head": [
                [-1, 1, "nn.Upsample", [None, 2, "nearest"]], [[-1, 6], 1, "Concat", [1]],
                [-1, 3, "C2f", [512]], [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
                [[-1, 4], 1, "Concat", [1]], [-1, 3, "C2f", [256]],
                [-1, 1, "Conv", [256, 3, 2]], [[-1, 12], 1, "Concat", [1]],
                [-1, 3, "C2f", [512]], [-1, 1, "Conv", [512, 3, 2]],
                [[-1, 9], 1, "Concat", [1]], [-1, 3, "C2f", [1024]],
                [[15, 18, 21], 1, "Detect", ["nc"]],
            ],
        }
    p3 = [-1, 3, "VSSBlock", [256]] if method == "Original" else [
        -1, 3, "DS_VSSBlockSweep", [256, 16, repr(control), {"trend_kernel_size": 3, "stats_kernel_size": 3}]]
    return {
        "nc": nc, "scales": {"T": [0.33, 0.25, 1024]}, "scale": "T",
        "backbone": [
            [-1, 1, "SimpleStem", [128, 3]], [-1, 3, "VSSBlock", [128]],
            [-1, 1, "VisionClueMerge", [256]], p3,
            [-1, 1, "VisionClueMerge", [512]], [-1, 9, "VSSBlock", [512]],
            [-1, 1, "VisionClueMerge", [1024]], [-1, 3, "VSSBlock", [1024]],
            [-1, 1, "SPPF", [1024, 5]],
        ],
        "head": [
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]], [[-1, 5], 1, "Concat", [1]],
            [-1, 3, "XSSBlock", [512]], [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 3], 1, "Concat", [1]], [-1, 3, "XSSBlock", [256]],
            [-1, 1, "Conv", [256, 3, 2]], [[-1, 11], 1, "Concat", [1]],
            [-1, 3, "XSSBlock", [512]], [-1, 1, "Conv", [512, 3, 2]],
            [[-1, 8], 1, "Concat", [1]], [-1, 3, "XSSBlock", [1024]],
            [[14, 17, 20], 1, "Detect", ["nc"]],
        ],
    }
