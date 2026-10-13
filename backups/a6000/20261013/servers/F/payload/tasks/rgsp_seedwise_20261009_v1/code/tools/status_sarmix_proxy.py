#!/usr/bin/env python3
"""Print compact progress for a split or complete SAR-Mix proxy queue."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


RUNS = (
    "baseline",
    "full_p2",
    "full_p3",
    "full_p4",
    "full_p5",
    "full_p2p3",
    "ds_only_p2",
    "pfim_only_p2",
    "global_p2",
)


def epoch_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--experiment-id", default="sarmix_scratch_d16_v1")
    args = parser.parse_args()

    project = Path(args.project).expanduser().resolve()
    print("run                 epochs  train  eval  AP50-95")
    print("------------------  ------  -----  ----  -------")
    completed = 0
    evaluated = 0
    for key in RUNS:
        run_dir = project / f"{args.experiment_id}_{key}"
        epochs = epoch_count(run_dir / "results.csv")
        train_done = (run_dir / "train_complete.json").exists()
        eval_path = run_dir / "eval_val_overall.json"
        ap = "-"
        if eval_path.exists():
            payload = json.loads(eval_path.read_text(encoding="utf-8"))
            ap = f"{float(payload['overall']['ap50_95']):.6f}"
            evaluated += 1
        completed += int(train_done)
        print(f"{key:18}  {epochs:6d}  {'yes' if train_done else 'no ':5}  {'yes' if eval_path.exists() else 'no ':4}  {ap}")
    print(f"\ntraining complete: {completed}/{len(RUNS)}")
    print(f"overall evaluated: {evaluated}/{len(RUNS)}")


if __name__ == "__main__":
    main()
