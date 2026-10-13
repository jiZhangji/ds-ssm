#!/usr/bin/env python3
"""Visualize DS-SSM corrections and PFIM guidance from a fixed checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pfim-weights", required=True)
    parser.add_argument("--baseline-weights", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--block-index", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    image_indices = [index for index, part in enumerate(parts) if part == "images"]
    if image_indices:
        parts[image_indices[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def names_dict(names) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return dict(enumerate(names))


def color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = (
        (230, 90, 40),
        (50, 180, 70),
        (40, 130, 230),
        (190, 90, 210),
        (30, 190, 190),
        (210, 150, 30),
        (140, 140, 140),
    )
    return palette[class_id % len(palette)]


def read_labels(image_path: Path) -> list[tuple[int, float, float, float, float]]:
    path = label_path(image_path)
    if not path.is_file():
        return []
    labels = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        labels.append((int(float(values[0])), *(float(value) for value in values[1:5])))
    return labels


def draw_ground_truth(
    image: np.ndarray,
    labels: list[tuple[int, float, float, float, float]],
    names: dict[int, str],
) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    for class_id, x, y, box_width, box_height in labels:
        x1 = int((x - box_width / 2) * width)
        y1 = int((y - box_height / 2) * height)
        x2 = int((x + box_width / 2) * width)
        y2 = int((y + box_height / 2) * height)
        color = color_for_class(class_id)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output,
            f"GT {names.get(class_id, class_id)}",
            (max(0, x1), max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


def draw_predictions(image: np.ndarray, result, names: dict[int, str]) -> np.ndarray:
    output = image.copy()
    if result.boxes is None:
        return output
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confidences = result.boxes.conf.detach().cpu().numpy()
    for box, class_id, confidence in zip(boxes, classes, confidences):
        x1, y1, x2, y2 = (int(value) for value in box)
        color = color_for_class(class_id)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output,
            f"{names.get(class_id, class_id)} {confidence:.2f}",
            (max(0, x1), max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


def title_panel(image: np.ndarray, title: str) -> np.ndarray:
    panel = np.full((image.shape[0] + 34, image.shape[1], 3), 255, dtype=np.uint8)
    panel[34:] = image
    cv2.putText(
        panel,
        title,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    return panel


def robust_bounds(*arrays: np.ndarray) -> tuple[float, float]:
    finite = np.concatenate([values[np.isfinite(values)].ravel() for values in arrays])
    if finite.size == 0:
        return 0.0, 1.0
    lower, upper = np.percentile(finite, (2.0, 98.0))
    if upper <= lower:
        lower, upper = float(finite.min()), float(finite.max())
    return float(lower), float(upper)


def robust_normalize(values: np.ndarray, bounds: tuple[float, float] | None = None) -> np.ndarray:
    lower, upper = bounds if bounds is not None else robust_bounds(values)
    return np.clip((values - lower) / max(upper - lower, 1e-8), 0.0, 1.0).astype(np.float32)


def align_map_to_image(values: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Remove letterbox padding in feature coordinates and restore the original aspect ratio."""
    image_height, image_width = image_shape
    feature_height, feature_width = values.shape
    scale = min(feature_width / image_width, feature_height / image_height)
    content_width = max(1, min(feature_width, int(round(image_width * scale))))
    content_height = max(1, min(feature_height, int(round(image_height * scale))))
    left = max(0, (feature_width - content_width) // 2)
    top = max(0, (feature_height - content_height) // 2)
    cropped = values[top : top + content_height, left : left + content_width]
    return cv2.resize(cropped, (image_width, image_height), interpolation=cv2.INTER_CUBIC)


def render_aligned_map(
    values: np.ndarray,
    bounds: tuple[float, float] | None = None,
) -> np.ndarray:
    normalized = robust_normalize(values, bounds=bounds)
    return cv2.applyColorMap(np.uint8(np.clip(normalized, 0, 1) * 255), cv2.COLORMAP_TURBO)


def overlay_map(image: np.ndarray, values: np.ndarray) -> np.ndarray:
    heatmap = render_aligned_map(values)
    return cv2.addWeighted(image, 0.58, heatmap, 0.42, 0)


def feature_mask(
    labels: list[tuple[int, float, float, float, float]],
    height: int,
    width: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for _, x, y, box_width, box_height in labels:
        x1 = max(0, min(width, int(math.floor((x - box_width / 2) * width))))
        y1 = max(0, min(height, int(math.floor((y - box_height / 2) * height))))
        x2 = max(0, min(width, int(math.ceil((x + box_width / 2) * width))))
        y2 = max(0, min(height, int(math.ceil((y + box_height / 2) * height))))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    return mask


def region_stats(values: np.ndarray, mask: np.ndarray, prefix: str) -> dict[str, float]:
    foreground = values[mask]
    background = values[~mask]
    fg_mean = float(foreground.mean()) if foreground.size else float("nan")
    bg_mean = float(background.mean()) if background.size else float("nan")
    return {
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_fg_mean": fg_mean,
        f"{prefix}_bg_mean": bg_mean,
        f"{prefix}_fg_minus_bg": fg_mean - bg_mean,
        f"{prefix}_fg_bg_ratio": fg_mean / bg_mean if bg_mean > 0 else float("nan"),
    }


class DSSSMCapture:
    def __init__(self, module) -> None:
        self.module = module
        self.values: dict[str, torch.Tensor | tuple[int, int]] = {}
        self.handles = [
            module.register_forward_pre_hook(self._capture_input, with_kwargs=True),
            module.pfim.register_forward_pre_hook(self._capture_pfim_input, with_kwargs=True),
            module.pfim.register_forward_hook(self._capture_importance),
        ]

    def clear(self) -> None:
        self.values.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def _capture_input(self, _module, args, kwargs) -> None:
        self.values["x"] = args[0].detach().float().cpu()
        hw_shape = kwargs.get("hw_shape")
        if hw_shape is not None:
            self.values["hw_shape"] = tuple(int(value) for value in hw_shape)

    def _capture_pfim_input(self, _module, args, kwargs) -> None:
        self.values["x_tilde"] = args[0].detach().float().cpu()
        self.values["x_bar"] = args[1].detach().float().cpu()
        hw_shape = kwargs.get("hw_shape")
        if hw_shape is not None:
            self.values["hw_shape"] = tuple(int(value) for value in hw_shape)

    def _capture_importance(self, _module, _args, output) -> None:
        self.values["importance"] = output.detach().float().cpu()

    def maps(self) -> dict[str, np.ndarray]:
        missing = {"x", "x_tilde", "x_bar", "importance", "hw_shape"} - self.values.keys()
        if missing:
            raise RuntimeError(f"DS-SSM capture is incomplete: {sorted(missing)}")
        x = self.values["x"][0]
        x_tilde = self.values["x_tilde"][0]
        x_bar = self.values["x_bar"][0]
        importance = self.values["importance"][0, :, 0]
        height, width = self.values["hw_shape"]
        if x.shape[0] != height * width:
            raise ValueError(f"token count {x.shape[0]} does not match feature shape {(height, width)}")

        input_energy = torch.sqrt(torch.mean(x.pow(2), dim=-1) + 1e-12)
        corrected_energy = torch.sqrt(torch.mean(x_tilde.pow(2), dim=-1) + 1e-12)
        residual = torch.sqrt(torch.mean((x - x_bar).pow(2), dim=-1) + 1e-12)
        correction = torch.sqrt(torch.mean((x - x_tilde).pow(2), dim=-1) + 1e-12)
        energy = torch.linalg.vector_norm(x_bar, dim=-1, keepdim=True) / math.sqrt(x_bar.shape[-1])
        centered = importance[:, None] - 0.5
        module = self.module
        gamma_logits = -module.alpha.detach().float().cpu() * module.sigma_n * energy
        if module.use_pfim and module.pfim_on_gamma:
            gamma_logits = gamma_logits + module.beta_i.detach().float().cpu() * centered
        gamma_raw = torch.sigmoid(gamma_logits)
        gamma = module.min_retain + (1.0 - module.min_retain) * gamma_raw
        if module.use_pfim and module.pfim_on_write:
            write_gate = 1.0 + torch.tanh(module.rho_i.detach().float().cpu()) * centered
            write_gate = torch.clamp(write_gate, min=0.1)
        else:
            write_gate = torch.ones_like(gamma)
        drive_gate = (gamma * write_gate)[:, 0]
        if module.use_pfim and module.pfim_on_delta:
            # Historical ex27 used centered importance. New sweep code exposes
            # ``pfim_tokenwise`` and uses the uncentered importance value.
            delta_signal = importance if hasattr(module, "pfim_tokenwise") else importance - 0.5
            delta_bias = module.eta_i.detach().float().cpu() * delta_signal
        else:
            delta_bias = torch.zeros_like(importance)

        return {
            "importance": importance.numpy().reshape(height, width),
            "input_energy": input_energy.numpy().reshape(height, width),
            "corrected_energy": corrected_energy.numpy().reshape(height, width),
            "residual": residual.numpy().reshape(height, width),
            "correction": correction.numpy().reshape(height, width),
            "drive_gate": drive_gate.numpy().reshape(height, width),
            "delta_bias": delta_bias.numpy().reshape(height, width),
        }


def find_dsssm_modules(model) -> list:
    return [
        module
        for module in model.modules()
        if type(module).__name__ == "DeSpeckleSSM" and getattr(module, "pfim", None) is not None
    ]


def predict(model: YOLO, image_path: Path, args: argparse.Namespace):
    return model.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        half=False,
        verbose=False,
    )[0]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = YOLO(args.baseline_weights)
    pfim = YOLO(args.pfim_weights)
    modules = find_dsssm_modules(pfim.model)
    if not modules:
        raise RuntimeError("The PFIM checkpoint contains no DeSpeckleSSM with PFIM")
    if not 0 <= args.block_index < len(modules):
        raise IndexError(f"block-index {args.block_index} is outside 0..{len(modules) - 1}")
    capture = DSSSMCapture(modules[args.block_index])
    names = names_dict(pfim.names)
    summaries = []

    try:
        for value in args.images:
            image_path = Path(value).expanduser().resolve()
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)
            labels = read_labels(image_path)
            baseline_result = predict(baseline, image_path, args)
            capture.clear()
            pfim_result = predict(pfim, image_path, args)
            maps = capture.maps()
            height, width = maps["importance"].shape
            aligned_maps = {
                name: align_map_to_image(values, image.shape[:2])
                for name, values in maps.items()
            }
            mask = feature_mask(labels, image.shape[0], image.shape[1])
            energy_bounds = robust_bounds(
                aligned_maps["input_energy"],
                aligned_maps["corrected_energy"],
            )

            panels = [
                title_panel(draw_ground_truth(image, labels, names), "Ground truth"),
                title_panel(draw_predictions(image, baseline_result, names), "Mamba-YOLO-T"),
                title_panel(draw_predictions(image, pfim_result, names), "DS-SSM + PFIM"),
                title_panel(overlay_map(image, aligned_maps["importance"]), "PFIM importance"),
                title_panel(
                    render_aligned_map(aligned_maps["input_energy"], bounds=energy_bounds),
                    "Input feature energy",
                ),
                title_panel(render_aligned_map(aligned_maps["residual"]), "Local residual"),
                title_panel(
                    render_aligned_map(aligned_maps["corrected_energy"], bounds=energy_bounds),
                    "Corrected feature energy",
                ),
                title_panel(render_aligned_map(aligned_maps["drive_gate"]), "Effective scan-drive gate"),
            ]
            top = cv2.hconcat(panels[:4])
            bottom = cv2.hconcat(panels[4:])
            figure = cv2.vconcat([top, bottom])
            output_path = output_dir / f"{image_path.stem}_dsssm_motivation.png"
            cv2.imwrite(str(output_path), figure)
            arrays = {f"raw_{name}": values for name, values in maps.items()}
            arrays.update({f"aligned_{name}": values for name, values in aligned_maps.items()})
            np.savez_compressed(output_dir / f"{image_path.stem}_dsssm_maps.npz", **arrays)

            summary = {
                "image": str(image_path),
                "output": str(output_path),
                "objects": len(labels),
                "feature_height": height,
                "feature_width": width,
                "baseline_predictions": len(baseline_result.boxes) if baseline_result.boxes is not None else 0,
                "pfim_predictions": len(pfim_result.boxes) if pfim_result.boxes is not None else 0,
            }
            for name, values in aligned_maps.items():
                summary.update(region_stats(values, mask, name))
            summaries.append(summary)
            print(output_path)
    finally:
        capture.close()

    if not summaries:
        raise RuntimeError("No images were processed")
    with (output_dir / "motivation_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    baseline_path = Path(args.baseline_weights).expanduser().resolve()
    pfim_path = Path(args.pfim_weights).expanduser().resolve()
    image_paths = [Path(value).expanduser().resolve() for value in args.images]
    manifest = {
        "tool": "visualize_dsssm_motivation.py",
        "tool_sha256": sha256(Path(__file__).resolve()),
        "baseline_weights": str(baseline_path),
        "baseline_weights_sha256": sha256(baseline_path),
        "pfim_weights": str(pfim_path),
        "pfim_weights_sha256": sha256(pfim_path),
        "imgsz": args.imgsz,
        "device": args.device,
        "conf": args.conf,
        "iou": args.iou,
        "block_index": args.block_index,
        "precision": "fp32",
        "images": [{"path": str(path), "sha256": sha256(path)} for path in image_paths],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"summary: {output_dir / 'motivation_summary.csv'}")
    print(f"manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
