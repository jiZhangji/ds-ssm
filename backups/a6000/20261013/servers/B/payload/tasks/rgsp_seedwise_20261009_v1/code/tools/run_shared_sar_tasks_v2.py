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
from ultralytics.nn.modules.block import SPPF  # noqa: E402
from ultralytics.nn.modules.vmamba_backbone import VMambaTinyBackbone  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_finite_loss(loss: torch.Tensor, dataset: str, variant: str, epoch: int, batch: int) -> None:
    if torch.isfinite(loss.detach()).all():
        return
    value = loss.detach().float().cpu().item() if loss.numel() == 1 else "non-scalar"
    raise FloatingPointError(
        f"Non-finite loss for dataset={dataset} variant={variant} epoch={epoch} batch={batch}: {value}"
    )


def require_finite_grad_norm(
    grad_norm: torch.Tensor, dataset: str, variant: str, epoch: int, batch: int
) -> None:
    if torch.isfinite(grad_norm.detach()).all():
        return
    value = grad_norm.detach().float().cpu().item() if grad_norm.numel() == 1 else "non-scalar"
    raise FloatingPointError(
        f"Non-finite gradient norm for dataset={dataset} variant={variant} "
        f"epoch={epoch} batch={batch}: {value}"
    )


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
    packed = (pixels[:, :, 0] << 16) | (pixels[:, :, 1] << 8) | pixels[:, :, 2]
    color_keys = (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]
    target = np.full(packed.shape, -1, dtype=np.int64)
    for index, key in enumerate(color_keys):
        target[packed == key] = index
    unmatched = target < 0
    if unmatched.any():
        values = pixels[unmatched]
        distance = ((values[:, None, :] - colors[None, :, :]) ** 2).sum(axis=2)
        target[unmatched] = distance.argmin(axis=1)
    return target


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
        fixed_angle = row.get("augmentation_angle")
        if fixed_angle is not None or (self.train and self.segmentation_augmentation):
            angles = self.segmentation_augmentation.get("angles", [0])
            angle = int(fixed_angle if fixed_angle is not None else angles[repeat_index % len(angles)])
            image = rotate_array(image, angle, nearest=False)
            mask = rotate_array(mask, angle, nearest=True)
            if self.train and self.segmentation_augmentation.get("random_horizontal_flip") and random.random() < 0.5:
                image, mask = np.flip(image, axis=1).copy(), np.flip(mask, axis=1).copy()
            if self.train and self.segmentation_augmentation.get("random_vertical_flip") and random.random() < 0.5:
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

    output_channels = (32, 128, 256)

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


class MambaYOLOTBackbone(nn.Module):
    """Mamba-YOLO-T backbone with the controlled replacement at true P3/8."""

    output_channels = (32, 64, 128, 256)

    def __init__(self, ours: bool, state_dim: int):
        super().__init__()
        self.stem = SimpleStem(3, 32, 3)
        self.p2 = VSSBlock(32, 32)
        self.merge2 = VisionClueMerge(32, 64)
        self.p3 = DS_VSSBlockSweep(64, 64, state_dim) if ours else VSSBlock(64, 64)
        self.merge3 = VisionClueMerge(64, 128)
        self.p4 = nn.Sequential(*(VSSBlock(128, 128) for _ in range(3)))
        self.merge4 = VisionClueMerge(128, 256)
        self.p5 = nn.Sequential(VSSBlock(256, 256), SPPF(256, 256, 5))

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = pad_to_stride(image, 32)
        p2 = self.p2(self.stem(image))
        p3 = self.p3(self.merge2(p2))
        p4 = self.p4(self.merge3(p3))
        p5 = self.p5(self.merge4(p4))
        return p2, p3, p4, p5


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


class PyramidPooling(nn.Module):
    def __init__(self, channels: int, width: int, scales: tuple[int, ...] = (1, 2, 3, 6)):
        super().__init__()
        branch_width = max(width // len(scales), 16)
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(channels, branch_width, 1, bias=False),
                nn.GroupNorm(1, branch_width),
                nn.SiLU(),
            )
            for scale in scales
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels + branch_width * len(scales), width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        size = value.shape[-2:]
        pooled = [
            F.interpolate(branch(value), size, mode="bilinear", align_corners=False)
            for branch in self.branches
        ]
        return self.fuse(torch.cat([value, *pooled], dim=1))


class UPerNetLiteDecoder(nn.Module):
    """Compact PPM+FPN decoder using P2-P5 features for fine SAR boundaries."""

    def __init__(self, channels: tuple[int, int, int, int], classes: int, width: int = 96):
        super().__init__()
        self.ppm = PyramidPooling(channels[-1], width)
        self.laterals = nn.ModuleList(nn.Conv2d(channel, width, 1) for channel in channels[:-1])
        self.smooth = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.SiLU(),
            )
            for _ in channels[:-1]
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(width * len(channels), width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(width, classes, 1),
        )

    def forward(self, features: tuple[torch.Tensor, ...], size: tuple[int, int]) -> torch.Tensor:
        top = self.ppm(features[-1])
        pyramid: list[torch.Tensor | None] = [None, None, None, top]
        for index in range(2, -1, -1):
            lateral = self.laterals[index](features[index])
            top = lateral + F.interpolate(top, lateral.shape[-2:], mode="bilinear", align_corners=False)
            top = self.smooth[index](top)
            pyramid[index] = top
        target = features[0].shape[-2:]
        fused = [
            value if value.shape[-2:] == target else F.interpolate(value, target, mode="bilinear", align_corners=False)
            for value in pyramid
            if value is not None
        ]
        return F.interpolate(self.classifier(torch.cat(fused, dim=1)), size, mode="bilinear", align_corners=False)


class FUSARMapASPP(nn.Module):
    """ASPP block matching the disclosed FUSAR-Map Net channel layout."""

    def __init__(self, channels: int, width: int = 256, rates: tuple[int, int, int] = (3, 6, 9)):
        super().__init__()
        self.rates = rates
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, width, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.rate_one = nn.Sequential(
            nn.Conv2d(channels, width, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.atrous = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, width, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            )
            for rate in rates
        )
        self.project = nn.Sequential(
            nn.Conv2d(channels + width * (2 + len(rates)), width, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        size = value.shape[-2:]
        pooled = F.interpolate(self.image_pool(value), size, mode="bilinear", align_corners=False)
        branches = [value, pooled, self.rate_one(value), *(branch(value) for branch in self.atrous)]
        return self.project(torch.cat(branches, dim=1))


class FUSARMapNetDecoder(nn.Module):
    """Paper-matched FUSAR-Map Net decoder adapted to Mamba-YOLO-T features.

    The public model uses a 256-channel ASPP output, 32-channel multiscale
    low-level projections, bilinear upsampling, and two 3x3 refinement layers.
    """

    def __init__(
        self,
        channels: tuple[int, int, int, int],
        classes: int,
        width: int = 256,
        low_level_width: int = 32,
    ):
        super().__init__()
        self.aspp = FUSARMapASPP(channels[2], width=width)
        self.low_level = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channel, low_level_width, 1, bias=False),
                nn.BatchNorm2d(low_level_width),
                nn.ReLU(inplace=True),
            )
            for channel in channels[:2]
        )
        fused_channels = width + low_level_width * len(self.low_level)
        self.refine = nn.Sequential(
            nn.Conv2d(fused_channels, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, classes, 1),
        )

    def forward(self, features: tuple[torch.Tensor, ...], size: tuple[int, int]) -> torch.Tensor:
        target = features[0].shape[-2:]
        high = F.interpolate(self.aspp(features[2]), target, mode="bilinear", align_corners=False)
        low = [
            projection(feature)
            for projection, feature in zip(self.low_level, features[:2])
        ]
        low = [
            value if value.shape[-2:] == target else F.interpolate(value, target, mode="bilinear", align_corners=False)
            for value in low
        ]
        logits = self.refine(torch.cat([high, *low], dim=1))
        return F.interpolate(logits, size, mode="bilinear", align_corners=False)


class SharedTaskModel(nn.Module):
    def __init__(
        self,
        task: str,
        classes: int,
        ours: bool,
        state_dim: int,
        backbone_layout: str = "legacy",
        decoder: str = "multiscale",
        pretrained: str | None = None,
    ):
        super().__init__()
        self.task = task
        if backbone_layout == "full_t":
            self.backbone = MambaYOLOTBackbone(ours, state_dim)
        elif backbone_layout == "vmamba_tiny":
            self.backbone = VMambaTinyBackbone(ours, state_dim, pretrained=pretrained)
        else:
            self.backbone = SharedBackbone(ours, state_dim)
        if task == "classification":
            feature_dim = int(self.backbone.output_channels[1])
            self.head = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(0.2),
                nn.Linear(feature_dim, classes),
            )
        else:
            if decoder == "upernet_lite":
                if backbone_layout not in ("full_t", "vmamba_tiny"):
                    raise ValueError("upernet_lite requires a full_t or vmamba_tiny backbone")
                self.decoder = UPerNetLiteDecoder(self.backbone.output_channels, classes)
            elif decoder == "fusar_map_net":
                if backbone_layout != "full_t":
                    raise ValueError("fusar_map_net requires backbone_layout=full_t")
                self.decoder = FUSARMapNetDecoder(self.backbone.output_channels, classes)
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
            # Classification must consume the P3 feature so the SAR-aware replacement
            # remains trainable during module warmup instead of becoming a dead branch.
            return self.head(features[1].mean(dim=(2, 3)))
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


def primary_score(task: str, metric: dict, manifest: dict, selection_metric: str | None = None) -> float:
    default = {"classification": "macro_f1", "segmentation": "miou_foreground", "change_detection": "f1"}[task]
    return float(metric[selection_metric or manifest.get("primary_metric", default)])


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
    segmentation_loss_override: str | None = None,
):
    if task == "classification":
        return lambda logits, target: F.cross_entropy(logits, target, label_smoothing=label_smoothing)
    if task == "segmentation":
        weights = segmentation_weights(manifest, output).to(device)
        ignore_index = int(manifest.get("ignore_index", -100))
        loss_name = segmentation_loss_override or manifest.get("segmentation_loss", "weighted_cross_entropy_plus_dice")

        def segmentation_loss(logits, target):
            if not bool((target != ignore_index).any()):
                return logits.sum() * 0.0
            cross_entropy = F.cross_entropy(
                logits, target, weight=weights, ignore_index=ignore_index
            )
            if loss_name == "weighted_cross_entropy":
                return cross_entropy
            return cross_entropy + multiclass_dice_loss(logits, target, ignore_index)

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


def load_matching_checkpoint(model: nn.Module, checkpoint_path: str | None) -> dict[str, object]:
    """Load only shape-compatible tensors and report exact transfer coverage."""
    if not checkpoint_path:
        return {"path": None, "loaded_tensors": 0, "model_tensors": len(model.state_dict()), "coverage": 0.0}
    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu", weights_only=False)
    source = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if hasattr(source, "state_dict"):
        source = source.state_dict()
    if not isinstance(source, dict):
        raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")
    target = model.state_dict()
    matched = {}
    for key, value in source.items():
        key = key.removeprefix("module.")
        if key in target and hasattr(value, "shape") and tuple(value.shape) == tuple(target[key].shape):
            matched[key] = value
    model.load_state_dict(matched, strict=False)
    return {
        "path": str(Path(checkpoint_path).expanduser().resolve()),
        "loaded_tensors": len(matched),
        "model_tensors": len(target),
        "coverage": len(matched) / max(len(target), 1),
    }


def configure_module_warmup(model: nn.Module, enabled: bool, train_decoder: bool = False) -> None:
    """Freeze the pretrained VMamba while fitting new P3 and optionally the task head."""
    if not enabled:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    p3 = getattr(model.backbone, "p3", None)
    if p3 is None or not any(parameter.numel() for parameter in p3.parameters()):
        raise RuntimeError("module warmup requires an ours model with trainable P3 parameters")
    for parameter in p3.parameters():
        parameter.requires_grad_(True)
    if train_decoder:
        decoder = getattr(model, "decoder", None)
        if decoder is None:
            raise RuntimeError("segmentation warmup requires a decoder")
        for parameter in decoder.parameters():
            parameter.requires_grad_(True)
    encoder = getattr(model.backbone, "encoder", None)
    if encoder is not None:
        encoder.eval()


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("no trainable parameters")
    if args.optimizer == "SGD":
        return torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.optimizer == "Adam":
        return torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)


def build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer, epochs: int):
    if args.scheduler == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma)
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    if args.scheduler == "fixed":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=args.lr * 0.01
    )


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


def parse_device_spec(spec: str) -> tuple[torch.device, list[int]]:
    """Parse a primary CUDA device or a comma-separated DataParallel device list."""
    values = [item.strip() for item in str(spec).split(",") if item.strip()]
    if len(values) == 1:
        device = torch.device(values[0])
        return device, [device.index] if device.type == "cuda" and device.index is not None else []
    if not values or any(not item.startswith("cuda:") for item in values):
        raise ValueError(f"multiple devices require cuda:N syntax, got {spec!r}")
    indices = [int(item.split(":", 1)[1]) for item in values]
    if len(set(indices)) != len(indices):
        raise ValueError(f"duplicate CUDA device in {spec!r}")
    available = torch.cuda.device_count()
    if any(index < 0 or index >= available for index in indices):
        raise ValueError(f"CUDA device list {indices} exceeds available device count {available}")
    return torch.device(f"cuda:{indices[0]}"), indices


def run(args) -> None:
    seed_everything(args.seed)
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task = manifest["task"]
    classes = len(manifest["classes"]) if task == "classification" else int(manifest.get("classes", 1))
    device, device_ids = parse_device_spec(args.device)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    if result_path.exists() and not args.force:
        print(f"SKIP COMPLETE: {result_path}")
        return

    core_model = SharedTaskModel(
        task,
        classes,
        args.variant == "ours",
        args.state_dim,
        backbone_layout=args.backbone_layout,
        decoder=args.decoder,
        pretrained=args.pretrained,
    ).to(device)
    model = (
        nn.DataParallel(core_model, device_ids=device_ids, output_device=device_ids[0])
        if len(device_ids) > 1
        else core_model
    )
    if args.pretrained and args.backbone_layout == "vmamba_tiny":
        transfer = getattr(core_model.backbone, "init_report", {
            "path": args.pretrained,
            "loaded_tensors": None,
            "model_tensors": len(core_model.state_dict()),
            "coverage": None,
        })
    elif args.pretrained:
        transfer = load_matching_checkpoint(core_model, args.pretrained)
    else:
        transfer = {"path": None, "loaded_tensors": 0, "model_tensors": len(core_model.state_dict()), "coverage": 0.0}
    if args.pretrained and args.backbone_layout != "vmamba_tiny":
        print(f"PRETRAIN TRANSFER: {transfer}", flush=True)
    module_warmup_epochs = (
        args.module_warmup_epochs
        if args.variant == "ours" and args.backbone_layout == "vmamba_tiny"
        else 0
    )
    initial_backbone = None
    if module_warmup_epochs:
        encoder = getattr(core_model.backbone, "encoder", None)
        if encoder is None:
            raise RuntimeError("module warmup requires a backbone encoder snapshot")
        initial_backbone = {
            key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()
        }
        configure_module_warmup(core_model, True, train_decoder=task == "segmentation")
    else:
        configure_module_warmup(core_model, False)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer, module_warmup_epochs or args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    criterion = build_loss(
        task,
        manifest,
        output,
        device,
        args.label_smoothing,
        segmentation_loss_override=args.segmentation_loss,
    )

    eval_imgsz = args.eval_imgsz or args.imgsz
    eval_batch = args.eval_batch or args.batch
    datasets = {
        split: ManifestDataset(
            manifest,
            split,
            args.imgsz if split == "train" else eval_imgsz,
            train=split == "train",
        )
        for split in ("train", "val", "test")
    }
    if len(datasets["train"]) == 0 or len(datasets["test"]) == 0:
        raise RuntimeError(f"Empty required split for {args.dataset}: { {key: len(value) for key, value in datasets.items()} }")
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch if split == "train" else eval_batch,
            shuffle=split == "train",
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        for split, dataset in datasets.items() if len(dataset)
    }
    has_validation = "val" in loaders
    grad_accum = max(int(args.grad_accum), 1)

    last_path, best_path = output / "last.pt", output / "best.pt"
    best_score, best_threshold, start_epoch, stale = -math.inf, None, 0, 0
    metric_best = {"miou_all": -math.inf, "fwiou": -math.inf}
    if last_path.exists() and not args.force:
        checkpoint = torch.load(last_path, map_location=device)
        core_model.load_state_dict(checkpoint["model"])
        best_score = checkpoint["best_score"]
        best_threshold = checkpoint.get("threshold")
        start_epoch = checkpoint["epoch"] + 1
        stale = checkpoint.get("stale", 0)
        metric_best.update(checkpoint.get("metric_best", {}))
        checkpoint_phase = checkpoint.get("phase", "full")
        if module_warmup_epochs and start_epoch >= module_warmup_epochs:
            if checkpoint_phase == "warmup" and initial_backbone is not None:
                core_model.backbone.encoder.load_state_dict(initial_backbone)
            configure_module_warmup(core_model, False)
            optimizer = build_optimizer(args, model)
            scheduler = build_scheduler(args, optimizer, args.epochs)
            best_score, best_threshold, stale = -math.inf, None, 0
            metric_best = {"miou_all": -math.inf, "fwiou": -math.inf}
        else:
            if module_warmup_epochs:
                configure_module_warmup(core_model, True, train_decoder=task == "segmentation")
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        print(f"RESUME: epoch {start_epoch}/{args.epochs}", flush=True)

    metrics_log = output / "metrics.jsonl"
    total_epochs = module_warmup_epochs + args.epochs
    for epoch in range(start_epoch, total_epochs):
        if module_warmup_epochs and epoch == module_warmup_epochs:
            if initial_backbone is None:
                raise RuntimeError("missing initial VMamba backbone snapshot")
            core_model.backbone.encoder.load_state_dict(initial_backbone)
            configure_module_warmup(core_model, False)
            optimizer = build_optimizer(args, model)
            scheduler = build_scheduler(args, optimizer, args.epochs)
            best_score, best_threshold, stale = -math.inf, None, 0
            metric_best = {"miou_all": -math.inf, "fwiou": -math.inf}
            print("WARMUP COMPLETE: restored ImageNet backbone and started full fine-tuning", flush=True)
        in_warmup = bool(module_warmup_epochs and epoch < module_warmup_epochs)
        model.train()
        if in_warmup:
            encoder = getattr(core_model.backbone, "encoder", None)
            if encoder is not None:
                encoder.eval()
        losses = []
        train_batches = len(loaders["train"])
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loaders["train"], start=1):
            inputs, target = move_batch(batch, device)
            if (batch_index - 1) % grad_accum == 0:
                group_size = min(grad_accum, train_batches - batch_index + 1)
            else:
                group_size = grad_accum
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                raw_loss = criterion(model(*inputs), target)
            require_finite_loss(raw_loss, args.dataset, args.variant, epoch + 1, batch_index)
            scaler.scale(raw_loss / group_size).backward()
            should_step = batch_index % grad_accum == 0 or batch_index == train_batches
            if should_step:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                require_finite_grad_norm(grad_norm, args.dataset, args.variant, epoch + 1, batch_index)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(raw_loss.detach()))
        scheduler.step()
        if has_validation:
            val_metric, threshold = evaluate(model, loaders["val"], task, classes, device)
            score = primary_score(task, val_metric, manifest, args.selection_metric)
            improved = (not in_warmup) and score > best_score
            stale = 0 if improved else stale + 1
        else:
            val_metric, threshold = {}, None
            score = -float(np.mean(losses))
            improved = True
            stale = 0
        if improved:
            best_score, best_threshold = score, threshold
        improved_metrics = []
        if task == "segmentation" and has_validation and not in_warmup:
            for metric_name, checkpoint_name in (("miou_all", "best_miou.pt"), ("fwiou", "best_fwiou.pt")):
                if float(val_metric[metric_name]) > metric_best[metric_name]:
                    metric_best[metric_name] = float(val_metric[metric_name])
                    improved_metrics.append(checkpoint_name)
        state = {
            "epoch": epoch,
            "model": core_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_score": best_score,
            "threshold": best_threshold,
            "stale": stale,
            "metric_best": metric_best.copy(),
            "phase": "warmup" if in_warmup else "full",
            "args": vars(args),
        }
        torch.save(state, last_path)
        if improved and not in_warmup:
            torch.save(state, best_path)
        for checkpoint_name in improved_metrics:
            torch.save(state, output / checkpoint_name)
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "lr": scheduler.get_last_lr()[0], "val": val_metric, "threshold": threshold}
        with metrics_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        if in_warmup:
            phase_progress = f"warmup {epoch + 1}/{module_warmup_epochs}"
        else:
            phase_progress = f"finetune {epoch - module_warmup_epochs + 1}/{args.epochs}"
        print(f"{args.dataset} {args.variant} {phase_progress} loss={row['loss']:.5f} score={score:.5f}", flush=True)
        if has_validation and not in_warmup and stale >= args.patience:
            print(f"EARLY STOP: {args.patience} stale epochs", flush=True)
            break

    checkpoint = torch.load(best_path, map_location=device)
    core_model.load_state_dict(checkpoint["model"])
    model.eval()
    test_metric, _ = evaluate(model, loaders["test"], task, classes, device, checkpoint.get("threshold"))
    params, gflops = profile_model(core_model, task, eval_imgsz, device)
    latency = latency_ms(core_model, task, eval_imgsz, device)
    result = {
        "dataset": args.dataset,
        "task": task,
        "variant": args.variant,
        "placement": "P3" if args.variant == "ours" else "baseline",
        "state_dim": args.state_dim if args.variant == "ours" else 0,
        "backbone_layout": args.backbone_layout,
        "decoder": args.decoder,
        "initialization": (
            "official_vmamba_tiny_backbone_plus_random_new_modules"
            if args.pretrained and args.backbone_layout == "vmamba_tiny"
            else "shape_matched_partial_transfer"
            if args.pretrained
            else "scratch"
        ),
        "pretrained": transfer,
        "protocol": manifest.get("protocol"),
        "selection_policy": manifest.get("selection_policy", "best-validation-score"),
        "class_names": manifest.get("class_names"),
        "split_sizes": {key: len(value) for key, value in datasets.items()},
        "imgsz": args.imgsz,
        "batch": args.batch,
        "devices": args.device,
        "num_devices": len(device_ids) or 1,
        "eval_imgsz": eval_imgsz,
        "eval_batch": eval_batch,
        "epochs_requested": args.epochs,
        "module_warmup_epochs": module_warmup_epochs,
        "fine_tune_epochs": args.epochs,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "lr": args.lr,
        "lr_milestones": args.lr_milestones,
        "lr_step_size": args.lr_step_size,
        "lr_gamma": args.lr_gamma,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "segmentation_loss": args.segmentation_loss or manifest.get("segmentation_loss"),
        "selection_metric": args.selection_metric or manifest.get("primary_metric"),
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
    parser.add_argument("--backbone-layout", choices=("legacy", "full_t", "vmamba_tiny"), default="legacy")
    parser.add_argument(
        "--decoder",
        choices=("multiscale", "upernet_lite", "fusar_map_net"),
        default="multiscale",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--module-warmup-epochs", type=int, default=0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--eval-batch", type=int)
    parser.add_argument("--eval-imgsz", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", choices=("Adam", "AdamW", "SGD"), default="AdamW")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler", choices=("cosine", "multistep", "step", "fixed"), default="cosine")
    parser.add_argument("--lr-milestones", nargs="*", type=int, default=[])
    parser.add_argument("--lr-step-size", type=int, default=2)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--segmentation-loss",
        choices=("weighted_cross_entropy", "weighted_cross_entropy_plus_dice"),
    )
    parser.add_argument("--selection-metric", choices=("miou_all", "miou_foreground", "fwiou", "oa"))
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--pretrained")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
