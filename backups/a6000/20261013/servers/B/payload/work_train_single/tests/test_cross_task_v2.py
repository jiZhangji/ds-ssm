from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_palette_mapping_uses_non_overflowing_distance():
    runner = load_module("run_shared_sar_tasks_v2", "tools/run_shared_sar_tasks_v2.py")
    palette = np.asarray([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]], dtype=np.uint8)
    mask = np.asarray([palette], dtype=np.uint8)
    assert runner.palette_to_index(mask, palette).tolist() == [[0, 1, 2, 3, 4]]


def test_segmentation_manifest_can_ignore_unknown_palette_pixels(tmp_path: Path):
    runner = load_module("run_shared_sar_tasks_v2_ignore", "tools/run_shared_sar_tasks_v2.py")
    image = tmp_path / "image.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (4, 4), (128, 128, 128)).save(image)
    target = np.zeros((4, 4, 3), dtype=np.uint8)
    target[:, 2:] = np.asarray([255, 0, 0], dtype=np.uint8)
    Image.fromarray(target).save(mask)
    manifest = {
        "task": "segmentation",
        "classes": 1,
        "palette": [[0, 0, 0], [255, 0, 0]],
        "label_map": [-1, 0],
        "ignore_index": -1,
        "records": [{"image": str(image), "mask": str(mask), "split": "train"}],
    }
    _, mapped = runner.ManifestDataset(manifest, "train", 4)[0]
    assert set(mapped.flatten().tolist()) == {-1, 0}
    assert int((mapped == -1).sum()) == 8


def test_multiclass_dice_ignores_unknown_pixels():
    runner = load_module("run_shared_sar_tasks_v2_dice", "tools/run_shared_sar_tasks_v2.py")
    target = torch.tensor([[[0, -1], [1, -1]]])
    first = torch.zeros((1, 2, 2, 2))
    second = first.clone()
    second[:, :, :, 1] = torch.tensor([[[100.0, -100.0], [-100.0, 100.0]]])
    assert torch.allclose(
        runner.multiclass_dice_loss(first, target, -1),
        runner.multiclass_dice_loss(second, target, -1),
    )


def test_segmentation_loss_is_finite_for_all_unknown_tile(tmp_path: Path):
    runner = load_module("run_shared_sar_tasks_v2_unknown_loss", "tools/run_shared_sar_tasks_v2.py")
    manifest = {
        "task": "segmentation",
        "classes": 2,
        "palette": [[0, 0, 0], [255, 0, 0], [0, 255, 0]],
        "label_map": [-1, 0, 1],
        "ignore_index": -1,
        "records": [],
    }
    output = tmp_path / "output"
    output.mkdir()
    (output / "class_weights.json").write_text("[1.0, 1.0]\n", encoding="utf-8")
    criterion = runner.build_loss("segmentation", manifest, output, torch.device("cpu"))
    logits = torch.randn(1, 2, 4, 4, requires_grad=True)
    loss = criterion(logits, torch.full((1, 4, 4), -1, dtype=torch.long))
    assert torch.isfinite(loss)
    assert float(loss) == 0.0


def test_non_finite_loss_is_rejected_before_backward():
    runner = load_module("run_shared_sar_tasks_v2_nonfinite_loss", "tools/run_shared_sar_tasks_v2.py")
    with pytest.raises(FloatingPointError, match=r"dataset=FUSAR-Map.*epoch=25.*batch=7"):
        runner.require_finite_loss(torch.tensor(float("nan")), "FUSAR-Map", "baseline", 25, 7)


def test_non_finite_gradient_norm_is_rejected_before_optimizer_step():
    runner = load_module("run_shared_sar_tasks_v2_nonfinite_grad", "tools/run_shared_sar_tasks_v2.py")
    with pytest.raises(FloatingPointError, match=r"variant=ours.*epoch=3.*batch=9"):
        runner.require_finite_grad_norm(torch.tensor(float("inf")), "FUSAR-Map", "ours", 3, 9)


def test_metrics_are_computed_from_global_confusion_counts():
    runner = load_module("run_shared_sar_tasks_v2_metrics", "tools/run_shared_sar_tasks_v2.py")
    confusion = torch.tensor([[8, 2], [1, 9]])
    metric = runner.classification_metrics(confusion)
    assert metric["accuracy"] == 0.85
    binary = runner.binary_metrics(tp=9, fp=2, fn=1, tn=8)
    assert abs(binary["f1"] - 18 / 21) < 1e-12
    assert "kappa" in binary


def test_classification_preparation_has_unique_class_ids(tmp_path: Path):
    prepare = load_module("prepare_cross_task_v2_classification", "tools/prepare_cross_task_v2.py")
    source = tmp_path / "source" / "dataset" / "images"
    for class_name in ("alpha", "beta"):
        directory = source / class_name
        directory.mkdir(parents=True)
        for index in range(10):
            Image.new("RGB", (8, 8), (index, 0, 0)).save(directory / f"{index}.png")
    archive = tmp_path / "classification.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in (tmp_path / "source").rglob("*"):
            if path.is_file():
                handle.write(path, path.relative_to(tmp_path / "source"))
    report = prepare.prepare_classification(archive, tmp_path / "prepared", seed=0)
    manifest = json.loads((tmp_path / "prepared" / "manifest.json").read_text(encoding="utf-8"))
    assert report["classes"] == ["alpha", "beta"]
    assert set(row["label"] for row in manifest["records"]) == {0, 1}
    assert report["split"] == {"train": 16, "val": 2, "test": 2}


def test_aconvnet_soc_transform_repeats_and_crops(tmp_path: Path):
    runner = load_module("run_shared_sar_tasks_v2_aconv", "tools/run_shared_sar_tasks_v2.py")
    image = tmp_path / "chip.jpeg"
    Image.fromarray(np.arange(128 * 128, dtype=np.uint8).reshape(128, 128)).save(image)
    manifest = {
        "task": "classification",
        "classes": ["target"],
        "records": [{"path": str(image), "label": 0, "split": "train"}],
        "classification_transform": {
            "name": "aconvnet-soc",
            "train_center_crop": 100,
            "generated_patch": 94,
            "model_crop": 88,
            "train_repeats": 49,
        },
    }
    dataset = runner.ManifestDataset(manifest, "train", 88, train=True)
    assert len(dataset) == 49
    sample, label = dataset[48]
    assert sample.shape == (3, 88, 88)
    assert int(label) == 0


def test_backbone_padding_keeps_paper_crop_and_aligns_network_stride():
    runner = load_module("run_shared_sar_tasks_v2_padding", "tools/run_shared_sar_tasks_v2.py")
    image = torch.randn(2, 3, 88, 88)
    padded = runner.pad_to_stride(image, 32)
    assert padded.shape == (2, 3, 96, 96)
    assert torch.equal(padded[:, :, :88, :88], image)
    aligned = torch.randn(1, 3, 256, 256)
    assert runner.pad_to_stride(aligned, 32) is aligned


def test_fusar_paper_augmentation_uses_all_fixed_angles(tmp_path: Path):
    runner = load_module("run_shared_sar_tasks_v2_fusar_paper", "tools/run_shared_sar_tasks_v2.py")
    image = tmp_path / "image.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (32, 32), (64, 64, 64)).save(image)
    target = np.zeros((32, 32, 3), dtype=np.uint8)
    target[8:24, 8:24] = np.asarray([255, 0, 0], dtype=np.uint8)
    Image.fromarray(target).save(mask)
    manifest = {
        "task": "segmentation",
        "classes": 4,
        "palette": [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]],
        "label_map": [1, 0, 1, 2, 3],
        "records": [{"image": str(image), "mask": str(mask), "split": "train"}],
        "segmentation_repeats": 8,
        "segmentation_augmentation": {
            "angles": [0, 45, 90, 135, 180, 225, 270, 315],
            "random_horizontal_flip": False,
            "random_vertical_flip": False,
        },
    }
    dataset = runner.ManifestDataset(manifest, "train", 32, train=True)
    assert len(dataset) == 8
    _, mapped = dataset[1]
    assert mapped.shape == (32, 32)
    assert set(mapped.unique().tolist()).issubset({0, 1, 2, 3})


def test_change_detection_patches_do_not_cross_spatial_groups(tmp_path: Path):
    prepare = load_module("prepare_cross_task_v2_change", "tools/prepare_cross_task_v2.py")
    source = tmp_path / "source"
    source.mkdir()
    for name in ("pair_1.bmp", "pair_2.bmp"):
        Image.new("L", (288, 288), 64).save(source / name)
    mask = np.zeros((288, 288), dtype=np.uint8)
    mask[40:250, 30:230] = 255
    Image.fromarray(mask).save(source / "pair_gt.bmp")
    archive = tmp_path / "pair.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in source.iterdir():
            handle.write(path, path.name)
    output = tmp_path / "prepared"
    prepare.prepare_change(tmp_path, output, archive.name, seed=0)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    groups = {"train": set(), "val": set(), "test": set()}
    for row in manifest["records"]:
        groups[row["split"]].add(row["group"])
        left, top, right, bottom = row["group_bounds"]
        x1, y1, x2, y2 = row["crop"]
        assert left <= x1 < x2 <= right
        assert top <= y1 < y2 <= bottom
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])
