#!/usr/bin/env python3
"""One val-only CUDA evaluator probe per approved CD dataset; no training/test."""
from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import cross_trial as trial
import manifest as campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    processes = subprocess.check_output([
        "nvidia-smi", "-i", args.gpu_uuid,
        "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader",
    ], text=True).strip()
    if processes:
        raise RuntimeError(f"Assigned GPU has an existing compute process; no preemption: {processes}")
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    base = trial.load_runtime(args.source_root)
    torch, np = base.torch, base.np
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = trial.create_model(base, "change_detection", 1, "original", 0).to(device).eval()
    Dataset = trial.dataset_class(base)
    rows = []
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("Ottawa", "Farmland", "Yellow_River"):
        path = Path(args.data_root).resolve() / name / "manifest.json"
        full_manifest = campaign.read_json(path)
        # Only val records are passed to the loader/evaluator. The shared scene
        # mask is decoded, but target pixels outside the val union never enter
        # the scores; this is checked against an independent coordinate union.
        val_records = [dict(row) for row in full_manifest["records"] if row["split"] == "val"]
        manifest = {**full_manifest, "records": val_records}
        dataset = Dataset(manifest, "val", 192, 0)
        if len(dataset) != len(val_records) or not val_records:
            raise RuntimeError("Unexpected validation record selection")
        scene_unions = {}
        for row in val_records:
            key = (row["image1"], row["image2"], row["mask"])
            if key not in scene_unions:
                with base.Image.open(row["mask"]) as image:
                    scene_unions[key] = np.zeros((image.height, image.width), dtype=bool)
            x1, y1, x2, y2 = row["crop"]
            scene_unions[key][y1:y2, x1:x2] = True
        expected_pixels = sum(int(union.sum()) for union in scene_unions.values())
        expected_positive = 0
        for key, union in scene_unions.items():
            with base.Image.open(key[2]) as image:
                expected_positive += int((np.asarray(image.convert("L"))[union] > 0).sum())
        metric, threshold = trial.evaluate(base, model, dataset, 16, 0, device, 0, 1)
        if metric["evaluated_native_unique_pixels"] != expected_pixels:
            raise RuntimeError("Evaluator counts do not equal the val coordinate union")
        if metric["tp"] + metric["fn"] != expected_positive:
            raise RuntimeError("Evaluator target positives differ from val-only target union")
        if sum(metric[key] for key in ("tp", "fp", "fn", "tn")) != expected_pixels:
            raise RuntimeError("Confusion counts do not partition the val coordinate union")
        if not math.isfinite(threshold) or not 0.10 <= threshold <= 0.90:
            raise RuntimeError("Validation threshold is invalid")
        for value in metric.values():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise RuntimeError("Nonfinite validation metric")
        rows.append({"dataset": name, "manifest_sha256": campaign.sha(path),
                     "validation_crop_count": len(val_records),
                     "naive_repeated_crop_pixels": sum((r["crop"][2]-r["crop"][0])*(r["crop"][3]-r["crop"][1]) for r in val_records),
                     "expected_unique_val_pixels": expected_pixels,
                     "expected_positive_val_pixels": expected_positive,
                     "threshold": threshold, "metric": metric,
                     "only_val_records_entered_loader": True,
                     "train_test_pixels_entering_metric": 0,
                     "shared_scene_mask_caveat": "The same source mask stores all splits; only the val coordinate union participates."})
    result = {"status": "passed", "validation_only": True, "formal_result": False,
              "model": "seed0 random Original full_t+UPerNet-lite, evaluation mode, no optimizer",
              "gpu_uuid": args.gpu_uuid, "source_root": str(Path(args.source_root).resolve()),
              "cross_trial_sha256": campaign.sha(Path(trial.__file__)),
              "probe_sha256": campaign.sha(Path(__file__)), "datasets": rows,
              "evaluations_per_dataset": 1, "test_evaluation_performed": False,
              "finished_utc": campaign.utc()}
    campaign.frozen(output / "result.json", result)
    print(__import__("json").dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
