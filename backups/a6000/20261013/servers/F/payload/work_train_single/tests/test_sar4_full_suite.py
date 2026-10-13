from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))


def load_runner():
    path = TOOLS / "run_sar4_full_suite.py"
    spec = importlib.util.spec_from_file_location("run_sar4_full_suite", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load_runner()


def test_full_p3_and_baseline_configs_have_expected_structure():
    baseline = RUNNER.model_config(nc=7, placement="baseline", state_dim=0, variant="baseline")
    full_p3 = RUNNER.model_config(nc=7, placement="p3", state_dim=16, variant="full")

    assert baseline["nc"] == 7
    assert baseline["head"][-1][3] == [7]
    assert all("DS_VSS" not in layer[2] for layer in baseline["backbone"])

    ds_layers = [layer for layer in full_p3["backbone"] if layer[2] == "DS_VSSBlockSweep"]
    assert len(ds_layers) == 1
    assert ds_layers[0][3] == [256, 16]
    assert full_p3["head"][-1][3] == [7]


def test_ssdd_split_is_frozen_disjoint_and_reproducible(tmp_path):
    dataset = tmp_path / "SSDD"
    train_dir = dataset / "images" / "train"
    test_dir = dataset / "images" / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    for index in range(928):
        (train_dir / f"train_{index:04d}.jpg").touch()
    for index in range(232):
        (test_dir / f"test_{index:04d}.jpg").touch()

    source = dataset / "SSDD.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "ship"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    generated = tmp_path / "manifest" / "datasets" / "SSDD.yaml"
    RUNNER.prepare_ssdd_split(source, generated, seed=0, val_count=186)

    payload = yaml.safe_load(generated.read_text(encoding="utf-8"))
    train = Path(payload["train"]).read_text(encoding="utf-8").splitlines()
    val = Path(payload["val"]).read_text(encoding="utf-8").splitlines()
    assert len(train) == 742
    assert len(val) == 186
    assert set(train).isdisjoint(val)
    assert payload["test"] == "images/test"

    RUNNER.prepare_ssdd_split(source, generated, seed=0, val_count=186)


def test_full_queue_dry_run_emits_ten_scratch_test_jobs(tmp_path, monkeypatch, capsys):
    data_root = tmp_path / "prepared"
    for spec in RUNNER.DATASETS:
        dataset = data_root / spec.folder
        dataset.mkdir(parents=True)
        payload = {"path": str(dataset), "train": "images/train", "val": "images/val", "test": "images/test"}
        payload["names"] = {0: "target"}
        (dataset / spec.yaml_name).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    ssdd = data_root / "SSDD" / "images"
    (ssdd / "train").mkdir(parents=True)
    (ssdd / "test").mkdir(parents=True)
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (ssdd / "train" / name).touch()
    (ssdd / "test" / "d.jpg").touch()

    project = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_sar4_full_suite.py",
            "--data-root",
            str(data_root),
            "--project",
            str(project),
            "--experiment-id",
            "dry_run_test",
            "--ssdd-val-count",
            "1",
            "--dry-run",
        ],
    )
    RUNNER.main()
    output = capsys.readouterr().out

    assert "Runs:       10" in output
    assert output.count("mbyolo_train_sarair.py") == 10
    assert output.count("evaluate_sarmix.py") == 10
    assert output.count("--weights none") == 10
    assert output.count("--module_warmup_epochs 0") == 10
    assert output.count("--split test") == 10


def test_selected_dataset_does_not_require_unselected_yamls(tmp_path, monkeypatch, capsys):
    data_root = tmp_path / "prepared"
    ssdd = data_root / "SSDD"
    (ssdd / "images" / "train").mkdir(parents=True)
    (ssdd / "images" / "test").mkdir(parents=True)
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (ssdd / "images" / "train" / name).touch()
    (ssdd / "images" / "test" / "d.jpg").touch()
    (ssdd / "SSDD.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(ssdd),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "ship"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_sar4_full_suite.py",
            "--data-root",
            str(data_root),
            "--project",
            str(tmp_path / "output"),
            "--experiment-id",
            "selected_only",
            "--datasets",
            "SSDD",
            "--variants",
            "baseline",
            "full_p3",
            "--ssdd-val-count",
            "1",
            "--dry-run",
        ],
    )
    RUNNER.main()
    output = capsys.readouterr().out
    assert "Runs:       2" in output
    assert "SSDD / baseline" in output
    assert "SSDD / full_p3" in output
