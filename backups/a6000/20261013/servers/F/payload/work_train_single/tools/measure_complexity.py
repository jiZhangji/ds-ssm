#!/usr/bin/env python3
"""Measure Params / GFLOPs / FPS for each variant's validation-selected checkpoint.

Loads best.pt via YOLO(), calls model.info(imgsz) for params/gflops, and reports
inference time already recorded in eval_test.json (speed_ms.inference).

Usage (on remote, from work_train_single):
  ../envs/ig-ds-ssm-env/bin/python tools/measure_complexity.py \
      --project ~/ig-ds-ssm/experiments/ssdd_paper_v1 \
      --experiment-ids seed0 seed1 seed2 --imgsz 640 --out /tmp/complexity.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parents[1] / "ultralytics").exists() else None
if ROOT:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import get_num_params, get_flops  # noqa: E402

VARIANTS = ("baseline", "full_p3", "ds_only_p3", "pfim_only_p3", "global_p3")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--experiment-ids", nargs="+", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    project = Path(args.project).expanduser().resolve()
    records = []

    for eid in args.experiment_ids:
        for variant in VARIANTS:
            run_dir = project / f"{eid}_ssdd_{variant}"
            best_pt = run_dir / "weights" / "best.pt"
            eval_json = run_dir / "eval_test.json"
            if not best_pt.exists():
                continue
            rec = {"experiment_id": eid, "variant": variant}
            # params (reliable) + gflops (thop may fail on DS-SSM selective_scan -> 0.0)
            try:
                model = YOLO(str(best_pt))
                params = int(get_num_params(model.model))
                rec["parameters"] = params
                rec["layers"] = len(list(model.model.modules()))
                gflops = get_flops(model.model, imgsz=args.imgsz)
                rec["gflops"] = float(gflops)
                rec["gflops_note"] = (
                    "thop could not trace the DS-SSM selective_scan custom op; "
                    "reported 0.0 is not a true value"
                    if gflops == 0.0
                    else "thop profile"
                )
            except Exception as exc:  # noqa: BLE001
                rec["info_error"] = f"{type(exc).__name__}: {exc}"
            # inference time from eval_test.json
            if eval_json.exists():
                item = json.loads(eval_json.read_text(encoding="utf-8"))
                inf_ms = item.get("speed_ms", {}).get("inference", 0.0)
                rec["inference_ms"] = float(inf_ms)
                rec["fps"] = round(1000.0 / float(inf_ms), 2) if inf_ms else None
                rec["ap50"] = item["overall"]["ap50"]
                rec["ap50_95"] = item["overall"]["ap50_95"]
            records.append(rec)

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2), encoding="utf-8")

    # print summary table
    print(f"{'variant':12} {'params':>12} {'gflops':>8} {'inf_ms':>8} {'fps':>8}")
    for rec in records:
        p = rec.get("parameters", "ERR")
        g = rec.get("gflops", "ERR")
        im = rec.get("inference_ms", "ERR")
        f = rec.get("fps", "ERR")
        print(f"{rec['experiment_id']}_{rec['variant']:12} {p:>12} {g:>8} {im:>8} {f:>8}")

    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
