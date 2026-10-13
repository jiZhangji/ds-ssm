#!/usr/bin/env python3
"""Controlled VSS/P3 runner for SAR classification, segmentation and change detection."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ultralytics.nn.modules.mamba_yolo import (  # noqa: E402
    DS_VSSBlockSweep,
    SimpleStem,
    VisionClueMerge,
    VSSBlock,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ImageCache:
    def __init__(self, limit: int = 8):
        self.limit = limit
        self.items: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, path: str, mode: str) -> np.ndarray:
        key = f"{path}:{mode}"
        if key not in self.items:
            array = np.asarray(Image.open(path).convert(mode)).copy()
            self.items[key] = array
            if len(self.items) > self.limit:
                self.items.popitem(last=False)
        return self.items[key].copy()


def resize_array(array: np.ndarray, size: int, nearest: bool = False) -> np.ndarray:
    mode = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return np.asarray(Image.fromarray(array).resize((size, size), mode)).copy()


def center_crop_array(array: np.ndarray, size: int) -> np.ndarray:
    height, width = array.shape[:2]
    if height < size or width < size:
        return resize_array(array, size)
    top = (height - size) // 2
    left = (width - size) // 2
    return array[top:top + size, left:left + size].copy()


def rotate_array(array: np.ndarray, angle: int, nearest: bool = False) -> np.ndarray:
    if angle % 360 == 0:
        return array.copy()
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return np.asarray(
        Image.fromarray(array).rotate(angle, resample=resample, expand=False, fillcolor=0)
    ).copy()


def paired_spatial_augment(arrays: list[np.ndarray]) -> list[np.ndarray]:
    if random.random() < 0.5:
        arrays = [np.flip(value, axis=1).copy() for value in arrays]
    if random.random() < 0.5:
        arrays = [np.flip(value, axis=0).copy() for value in arrays]
    turns = random.randrange(4)
    if turns:
        arrays = [np.rot90(value, turns, axes=(0, 1)).copy() for value in arrays]
    return arrays


def image_tensor(array: np.ndarray) -> torch.Tensor:
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    value = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
    return (value - 0.5) / 0.5


def palette_to_index(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    pixels = mask.astype(np.int32)
    colors = palette.astype(np.int32)
    distance = ((pixels[:, :, None, :] - colors[None, None, :, :]) ** 2).sum(axis=3)
    return distance.argmin(axis=2).astype(np.int64)


class ManifestDataset(Dataset):
    def __init__(self, manifest: dict, split: str, size: int, train: bool = False):
        self.task = manifest["task"]
        self.records = [row for row in manifest["records"] if row.get("split") == split]
        self.size = size
        self.train = train
        self.palette = np.asarray(manifest.get("palette", []), dtype=np.uint8)
        self.label_map = np.asarray(manifest.get("label_map", []), dtype=np.int64)
        self.ignore_index = int(manifest.get("ignore_index", -1))
        self.classification_transform = manifest.get("classification_transform", {})
        self.segmentation_augmentation = manifest.get("segmentation_augmentation", {})
        self.repeats = 1
        if train and self.task == "classification":
            self.repeats = int(self.classification_transform.get("train_repeats", 1))
        if train and self.task == "segmentation":
            self.repeats = int(manifest.get("segmentation_repeats", 1))
        self.cache = ImageCache()

    def __len__(self) -> int:
        return len(self.records) * self.repeats

    def classification(self, row: dict, repeat_index: int):
        image = self.cache.get(row["path"], "RGB")
        if self.classification_transform.get("name") == "aconvnet-soc":
            model_crop = int(self.classification_transform.get("model_crop", 88))
            if self.train:
                center_size = int(self.classification_transform.get("train_center_crop", 100))
                generated_patch = int(self.classification_transform.get("generated_patch", 94))
                image = center_crop_array(image, center_size)
                positions = center_size - generated_patch + 1
                top = (repeat_index // positions) % positions
                left = repeat_index % positions
                image = image[top:top + generated_patch, left:left + generated_patch]
                maximum = generated_patch - model_crop
                top = random.randint(0, maximum)
                left = random.randint(0, maximum)
                image = image[top:top + model_crop, left:left + model_crop]
            else:
                image = center_crop_array(image, int(self.classification_transform.get("test_crop", model_crop)))
            if image.shape[0] != self.size or image.shape[1] != self.size:
                image = resize_array(image, self.size)
        elif self.train:
            image = resize_array(image, round(self.size * 1.12))
            maximum = image.shape[0] - self.size
            top = random.randint(0, max(maximum, 0))
            maximum = image.shape[1] - self.size
            left = random.randint(0, max(maximum, 0))
            image = image[top:top + self.size, left:left + self.size]
            image = paired_spatial_augment([image])[0]
        else:
            image = resize_array(image, self.size)
        return image_tensor(image), torch.tensor(row["label"], dtype=torch.long)

    def segmentation(self, row: dict, repeat_index: int):
        image = self.cache.get(row["image"], "RGB")
        mask = self.cache.get(row["mask"], "RGB")
        if "crop" in row:
            left, top, right, bottom = row["crop"]
            image = image[top:bottom, left:right]
            mask = mask[top:bottom, left:right]
        if self.train and min(image.shape[:2]) > self.size:
            top = random.randint(0, image.shape[0] - self.size)
            left = random.randint(0, image.shape[1] - self.size)
            image = image[top:top + self.size, left:left + self.size]
            mask = mask[top:top + self.size, left:left + self.size]
        else:
            image = resize_array(image, self.size)
            mask = resize_array(mask, self.size, nearest=True)
        if self.train and self.segmentation_augmentation:
            angles = self.segmentation_augmentation.get("angles", [0])
            angle = int(angles[repeat_index % len(angles)])
            image = rotate_array(image, angle, nearest=False)
            mask = rotate_array(mask, angle, nearest=True)
            if self.segmentation_augmentation.get("random_horizontal_flip") and random.random() < 0.5:
                image, mask = np.flip(image, axis=1).copy(), np.flip(mask, axis=1).copy()
            if self.segmentation_augmentation.get("random_vertical_flip") and random.random() < 0.5:
                image, mask = np.flip(image, axis=0).copy(), np.flip(mask, axis=0).copy()
        elif self.train:
            image, mask = paired_spatial_augment([image, mask])
        target = palette_to_index(mask, self.palette)
        if self.label_map.size:
            mapped = np.full(target.shape, self.ignore_index, dtype=np.int64)
            valid = target < len(self.label_map)
            mapped[valid] = self.label_map[target[valid]]
            target = mapped
        return image_tensor(image), torch.from_numpy(target)

    def change_detection(self, row: dict):
        crop = tuple(row["crop"])
        image1 = self.cache.get(row["image1"], "RGB")[crop[1]:crop[3], crop[0]:crop[2]]
        image2 = self.cache.get(row["image2"], "RGB")[crop[1]:crop[3], crop[0]:crop[2]]
        mask = self.cache.get(row["mask"], "L")[crop[1]:crop[3], crop[0]:crop[2]]
        image1 = resize_array(image1, self.size)
        image2 = resize_array(image2, self.size)
        mask = resize_array(mask, self.size, nearest=True)
        if self.train:
            image1, image2, mask = paired_spatial_augment([image1, image2, mask])
        target = torch.from_numpy((mask > 0).astype(np.float32))[None]
        return (image_tensor(image1), image_tensor(image2)), target

    def __getitem__(self, index: int):
        row = self.records[index // self.repeats]
        repeat_index = index % self.repeats
        if self.task == "classification":
            return self.classification(row, repeat_index)
        if self.task == "segmentation":
            return self.segmentation(row, repeat_index)
        return self.change_detection(row)


class SharedBackbone(nn.Module):
    """Mamba-YOLO-style VSS backbone with the controlled replacement at P3."""

    def __init__(self, ours: bool, state_dim: int):
        super().__init__()
        self.stem = SimpleStem(3, 32, 3)
        self.p2 = VSSBlock(32, 32)
        self.merge2 = VisionClueMerge(32, 64)
        self.p3_entry = VSSBlock(64, 64)
        self.merge3 = VisionClueMerge(64, 128)
        self.p3 = DS_VSSBlockSweep(128, 128, state_dim) if ours else VSSBlock(128, 128)
        self.merge4 = VisionClueMerge(128, 256)
        self.p4 = VSSBlock(256, 256)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = pad_to_stride(image, 32)
        p2 = self.p2(self.stem(image))
        x = self.p3_entry(self.merge2(p2))
        p3 = self.p3(self.merge3(x))
        p4 = self.p4(self.merge4(p3))
        return p2, p3, p4


def pad_to_stride(image: torch.Tensor, stride: int) -> torch.Tensor:
    """Pad the bottom/right edges so multiscale merges remain shape-compatible."""
    height, width = image.shape[-2:]
    pad_height = (-height) % stride
    pad_width = (-width) % stride
    if not pad_height and not pad_width:
        return image
    return F.pad(image, (0, pad_width, 0, pad_height))


class MultiScaleDecoder(nn.Module):
    def __init__(self, classes: int, width: int = 96):
        super().__init__()
        self.l2 = nn.Conv2d(32, width, 1)
        self.l3 = nn.Conv2d(128, width, 1)
        self.l4 = nn.Conv2d(256, width, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(width * 3, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(),
            nn.Conv2d(width, classes, 1),
        )

    def forward(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor], size: tuple[int, int]):
        p2, p3, p4 = features
        target = p2.shape[-2:]
        values = [
            self.l2(p2),
            F.interpolate(self.l3(p3), target, mode="bilinear", align_corners=False),
            F.interpolate(self.l4(p4), target, mode="bilinear", align_corners=False),
        ]
        return F.interpolate(self.fuse(torch.cat(values, dim=1)), size, mode="bilinear", align_corners=False)


class SharedTaskModel(nn.Module):
    def __init__(self, task: str, classes: int, ours: bool, state_dim: int):
        super().__init__()
        self.task = task
        self.backbone = SharedBackbone(ours, state_dim)
        if task == "classification":
            self.head = nn.Sequential(nn.LayerNorm(256), nn.Dropout(0.2), nn.Linear(256, classes))
        else:
            self.decoder = MultiScaleDecoder(classes)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if self.task == "change_detection":
            first = self.backbone(inputs[0])
            second = self.backbone(inputs[1])
            difference = tuple((a - b).abs() for a, b in zip(first, second))
            return self.decoder(difference, inputs[0].shape[-2:])
        features = self.backbone(inputs[0])
        if self.task == "classification":
            return self.head(features[-1].mean(dim=(2, 3)))
        return self.decoder(features, inputs[0].shape[-2:])


def multiclass_dice_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int | None = None) -> torch.Tensor:
    classes = logits.shape[1]
    valid = torch.ones_like(target, dtype=torch.bool) if ignore_index is None else target != ignore_index
    safe_target = target.masked_fill(~valid, 0)
    one_hot = F.one_hot(safe_target, classes).permute(0, 3, 1, 2).float()
    valid = valid[:, None].float()
    one_hot = one_hot * valid
    probability = logits.softmax(dim=1)
    probability = probability * valid
    intersection = (probability * one_hot).sum(dim=(0, 2, 3))
    denominator = (probability + one_hot).sum(dim=(0, 2, 3))
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def binary_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = logits.sigmoid()
    intersection = (probability * target).sum(dim=(0, 2, 3))
    denominator = (probability + target).sum(dim=(0, 2, 3))
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def confusion_from_predictions(prediction: torch.Tensor, target: torch.Tensor, classes: int) -> torch.Tensor:
    valid = (target >= 0) & (target < classes)
    index = classes * target[valid].long() + prediction[valid].long()
    return torch.bincount(index, minlength=classes * classes).reshape(classes, classes).cpu()


def classification_metrics(confusion: torch.Tensor) -> dict[str, float]:
    value = confusion.double()
    tp = value.diag()
    precision = tp / value.sum(0).clamp_min(1)
    recall = tp / value.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "accuracy": float(tp.sum() / value.sum().clamp_min(1)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class_recall": [float(item) for item in recall],
        "confusion_matrix": confusion.tolist(),
    }


def segmentation_metrics(confusion: torch.Tensor) -> dict[str, float | list[float]]:
    value = confusion.double()
    tp = value.diag()
    ground = value.sum(1)
    predicted = value.sum(0)
    union = ground + predicted - tp
    iou = tp / union.clamp_min(1)
    pa = tp / ground.clamp_min(1)
    dice = 2 * tp / (ground + predicted).clamp_min(1)
    frequency = ground / ground.sum().clamp_min(1)
    return {
        "miou_all": float(iou.mean()),
        "miou_foreground": float(iou[1:].mean()),
        "mpa": float(pa.mean()),
        "dice": float(dice.mean()),
        "oa": float(tp.sum() / value.sum().clamp_min(1)),
        "fwiou": float((frequency * iou).sum()),
        "per_class_iou": [float(item) for item in iou],
        "per_class_pa": [float(item) for item in pa],
        "per_class_dice": [float(item) for item in dice],
    }


def binary_metrics(tp: float, fp: float, fn: float, tn: float) -> dict[str, float]:
    total = tp + fp + fn + tn
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    oa = (tp + tn) / max(total, 1e-12)
    expected = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / max(total * total, 1e-12)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "iou": tp / max(tp + fp + fn, 1e-12),
        "oa": oa,
        "kappa": (oa - expected) / max(1 - expected, 1e-12),
    }


def move_batch(batch, device: torch.device):
    inputs, target = batch
    if isinstance(inputs, (tuple, list)):
        inputs = tuple(value.to(device, non_blocking=True) for value in inputs)
    else:
        inputs = (inputs.to(device, non_blocking=True),)
    return inputs, target.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model, loader, task: str, classes: int, device: torch.device, threshold: float | None = None):
    model.eval()
    if task in {"classification", "segmentation"}:
        confusion = torch.zeros((classes, classes), dtype=torch.int64)
        for batch in loader:
            inputs, target = move_batch(batch, device)
            prediction = model(*inputs).argmax(1)
            confusion += confusion_from_predictions(prediction, target, classes)
        metrics = classification_metrics(confusion) if task == "classification" else segmentation_metrics(confusion)
        return metrics, threshold

    probabilities, targets = [], []
    for batch in loader:
        inputs, target = move_batch(batch, device)
        probabilities.append(model(*inputs).sigmoid().flatten().cpu())
        targets.append(target.flatten().cpu().bool())
    probability = torch.cat(probabilities)
    target = torch.cat(targets)
    thresholds = [threshold] if threshold is not None else np.arange(0.10, 0.91, 0.05)
    best_metric, best_threshold = None, None
    for candidate in thresholds:
        prediction = probability >= float(candidate)
        tp = float((prediction & target).sum())
        fp = float((prediction & ~target).sum())
        fn = float((~prediction & target).sum())
        tn = float((~prediction & ~target).sum())
        metric = binary_metrics(tp, fp, fn, tn)
        if best_metric is None or metric["f1"] > best_metric["f1"]:
            best_metric, best_threshold = metric, float(candidate)
    return best_metric, best_threshold


def primary_score(task: str, metric: dict, manifest: dict) -> float:
    default = {"classification": "macro_f1", "segmentation": "miou_foreground", "change_detection": "f1"}[task]
    return float(metric[manifest.get("primary_metric", default)])


def segmentation_weights(manifest: dict, output: Path) -> torch.Tensor:
    weight_cache = output / "class_weights.json"
    if weight_cache.exists():
        return torch.tensor(json.loads(weight_cache.read_text(encoding="utf-8")), dtype=torch.float32)
    palette = np.asarray(manifest["palette"], dtype=np.uint8)
    classes = int(manifest["classes"])
    label_map = np.asarray(manifest.get("label_map", []), dtype=np.int64)
    ignore_index = int(manifest.get("ignore_index", -1))
    counts = np.zeros(classes, dtype=np.float64)
    image_cache = ImageCache(limit=4)
    for row in manifest["records"]:
        if row["split"] != "train":
            continue
        mask = image_cache.get(row["mask"], "RGB")
        if "crop" in row:
            left, top, right, bottom = row["crop"]
            mask = mask[top:bottom, left:right]
        target = palette_to_index(mask, palette)
        if label_map.size:
            target = label_map[target]
        target = target[target != ignore_index]
        counts += np.bincount(target.ravel(), minlength=classes)[:classes]
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    weights = np.clip(weights / weights.mean(), 0.25, 4.0)
    weight_cache.write_text(json.dumps(weights.tolist(), indent=2) + "\n", encoding="utf-8")
    return torch.tensor(weights, dtype=torch.float32)


def change_positive_weight(manifest: dict) -> float:
    positive = total = 0
    cache = ImageCache(limit=4)
    for row in manifest["records"]:
        if row["split"] != "train":
            continue
        x1, y1, x2, y2 = row["crop"]
        mask = cache.get(row["mask"], "L")[y1:y2, x1:x2] > 0
        positive += int(mask.sum())
        total += mask.size
    return min(max((total - positive) / max(positive, 1), 1.0), 20.0)


def build_loss(
    task: str,
    manifest: dict,
    output: Path,
    device: torch.device,
    label_smoothing: float = 0.1,
):
    if task == "classification":
        return lambda logits, target: F.cross_entropy(logits, target, label_smoothing=label_smoothing)
    if task == "segmentation":
        weights = segmentation_weights(manifest, output).to(device)
        ignore_index = int(manifest.get("ignore_index", -100))

        def segmentation_loss(logits, target):
            if not bool((target != ignore_index).any()):
                return logits.sum() * 0.0
            return F.cross_entropy(
                logits, target, weight=weights, ignore_index=ignore_index
            ) + multiclass_dice_loss(logits, target, ignore_index)

        return segmentation_loss
    weight = torch.tensor(change_positive_weight(manifest), device=device)
    return lambda logits, target: F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight) + binary_dice_loss(logits, target)


def profile_model(model: nn.Module, task: str, size: int, device: torch.device) -> tuple[int, float | None]:
    params = sum(parameter.numel() for parameter in model.parameters())
    try:
        from thop import profile
        sample = torch.zeros(1, 3, size, size, device=device)
        inputs = (sample, sample.clone()) if task == "change_detection" else (sample,)
        flops, _ = profile(model, inputs=inputs, verbose=False)
        return params, float(flops / 1e9)
    except Exception:
        return params, None


@torch.no_grad()
def latency_ms(model: nn.Module, task: str, size: int, device: torch.device) -> float:
    sample = torch.zeros(1, 3, size, size, device=device)
    inputs = (sample, sample.clone()) if task == "change_detection" else (sample,)
    for _ in range(10):
        model(*inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    begin = time.perf_counter()
    for _ in range(30):
        model(*inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - begin) * 1000.0 / 30.0


def run(args) -> None:
    seed_everything(args.seed)
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task = manifest["task"]
    classes = len(manifest["classes"]) if task == "classification" else int(manifest.get("classes", 1))
    device = torch.device(args.device)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    if result_path.exists() and not args.force:
        print(f"SKIP COMPLETE: {result_path}")
        return

    model = SharedTaskModel(task, classes, args.variant == "ours", args.state_dim).to(device)
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=args.lr_milestones,
            gamma=args.lr_gamma,
        )
    elif args.scheduler == "fixed":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.epochs, 1),
            eta_min=args.lr * 0.01,
        )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    criterion = build_loss(task, manifest, output, device, args.label_smoothing)

    datasets = {
        split: ManifestDataset(manifest, split, args.imgsz, train=split == "train")
        for split in ("train", "val", "test")
    }
    if len(datasets["train"]) == 0 or len(datasets["test"]) == 0:
        raise RuntimeError(f"Empty required split for {args.dataset}: { {key: len(value) for key, value in datasets.items()} }")
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch,
            shuffle=split == "train",
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        for split, dataset in datasets.items() if len(dataset)
    }
    has_validation = "val" in loaders

    last_path, best_path = output / "last.pt", output / "best.pt"
    best_score, best_threshold, start_epoch, stale = -math.inf, None, 0, 0
    if last_path.exists() and not args.force:
        checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        best_score = checkpoint["best_score"]
        best_threshold = checkpoint.get("threshold")
        start_epoch = checkpoint["epoch"] + 1
        stale = checkpoint.get("stale", 0)
        print(f"RESUME: epoch {start_epoch}/{args.epochs}", flush=True)

    metrics_log = output / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        for batch in loaders["train"]:
            inputs, target = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                loss = criterion(model(*inputs), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()
        if has_validation:
            val_metric, threshold = evaluate(model, loaders["val"], task, classes, device)
            score = primary_score(task, val_metric, manifest)
            improved = score > best_score
            stale = 0 if improved else stale + 1
        else:
            val_metric, threshold = {}, None
            score = -float(np.mean(losses))
            improved = True
            stale = 0
        if improved:
            best_score, best_threshold = score, threshold
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_score": best_score,
            "threshold": best_threshold,
            "stale": stale,
            "args": vars(args),
        }
        torch.save(state, last_path)
        if improved:
            torch.save(state, best_path)
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "lr": scheduler.get_last_lr()[0], "val": val_metric, "threshold": threshold}
        with metrics_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(f"{args.dataset} {args.variant} epoch={epoch + 1}/{args.epochs} loss={row['loss']:.5f} score={score:.5f}", flush=True)
        if has_validation and stale >= args.patience:
            print(f"EARLY STOP: {args.patience} stale epochs", flush=True)
            break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    test_metric, _ = evaluate(model, loaders["test"], task, classes, device, checkpoint.get("threshold"))
    params, gflops = profile_model(model, task, args.imgsz, device)
    latency = latency_ms(model, task, args.imgsz, device)
    result = {
        "dataset": args.dataset,
        "task": task,
        "variant": args.variant,
        "placement": "P3" if args.variant == "ours" else "baseline",
        "state_dim": args.state_dim if args.variant == "ours" else 0,
        "initialization": "scratch",
        "protocol": manifest.get("protocol"),
        "selection_policy": manifest.get("selection_policy", "best-validation-score"),
        "class_names": manifest.get("class_names"),
        "split_sizes": {key: len(value) for key, value in datasets.items()},
        "imgsz": args.imgsz,
        "batch": args.batch,
        "epochs_requested": args.epochs,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "lr": args.lr,
        "lr_milestones": args.lr_milestones,
        "lr_gamma": args.lr_gamma,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "best_epoch": checkpoint["epoch"] + 1,
        "threshold": checkpoint.get("threshold"),
        "metric": test_metric,
        "params": params,
        "gflops": gflops,
        "latency_ms": latency,
        "fps": 1000.0 / latency,
        "checkpoint": str(best_path),
        "manifest": str(manifest_path),
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", choices=("baseline", "ours"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", choices=("Adam", "AdamW", "SGD"), default="AdamW")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler", choices=("cosine", "multistep", "fixed"), default="cosine")
    parser.add_argument("--lr-milestones", nargs="*", type=int, default=[])
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
