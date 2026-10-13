from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_prepare_module():
    path = ROOT / "tools" / "prepare_cross_task_data.py"
    spec = importlib.util.spec_from_file_location("prepare_cross_task_data", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_authoritative_matrix_contains_all_twelve_datasets():
    module = load_prepare_module()
    assert len(module.DETECTION) == 5
    assert len(module.CLASSIFICATION) == 2
    assert len(module.SEGMENTATION) == 1
    assert len(module.CHANGE) == 4
    assert len(module.ALL) == 12
    assert "OGSOD" not in module.ALL
    assert "SAR-Ship-Dataset" not in module.ALL
    assert set(module.DATASET_DIRS) == set(module.ALL)


def test_shared_run_count_is_ten_detection_plus_fourteen_shared():
    script = (ROOT / "tools" / "run_all_cross_task.sh").read_text(encoding="utf-8")
    assert "Planned runs: 24" in script
    assert "DETECTION_DATASETS=(SSDD SAR-AIRcraft1.0 AIR-SARShip-2.0 SRSDD-v1.0 SADD)" in script
    assert 'SHARED_DATASETS=(SAR-ACD SAMPLE FUSAR-Map Ottawa "Yellow River" Farmland Sulzberger)' in script
