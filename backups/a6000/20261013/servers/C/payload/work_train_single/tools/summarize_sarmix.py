#!/usr/bin/env python3
"""Aggregate SAR-Mix validation metrics into CSV, JSON, and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "experiment",
    "variant",
    "placement",
    "state_dim",
    "seed",
    "subset",
    "precision",
    "recall",
    "ap50",
    "ap75",
    "ap50_95",
    "inference_ms",
    "weights",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--experiment-id", required=True)
    return parser.parse_args()


def load_records(project: Path, experiment_id: str) -> list[dict]:
    records = []
    for path in sorted(project.glob(f"{experiment_id}_*/eval_val_*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if item.get("experiment", "").startswith(experiment_id):
            records.append(item)
    return records


def row(item: dict) -> dict:
    return {
        "experiment": item["experiment"],
        "variant": item["variant"],
        "placement": item["placement"],
        "state_dim": item["state_dim"],
        "seed": item["seed"],
        "subset": item["subset"],
        **item["overall"],
        "inference_ms": item.get("speed_ms", {}).get("inference", ""),
        "weights": item["weights"],
    }


def main() -> None:
    args = parse_args()
    project = Path(args.project).expanduser().resolve()
    records = load_records(project, args.experiment_id)
    if not records:
        print(f"No evaluation JSON found for {args.experiment_id} under {project}")
        return

    rows = [row(item) for item in records]
    rows.sort(key=lambda item: (item["subset"] != "overall", -item["ap50_95"], item["experiment"]))
    csv_path = project / f"{args.experiment_id}_all_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    overall = [item for item in rows if item["subset"] == "overall"]
    best_placement = next(
        (
            item
            for item in overall
            if item["variant"] == "full" and item["placement"] in {"p2", "p3", "p4", "p5", "p2p3"}
        ),
        None,
    )
    payload = {
        "experiment_id": args.experiment_id,
        "completed_experiments": len(overall),
        "best_overall": overall[0] if overall else None,
        "best_full_placement": best_placement,
        "records": rows,
    }
    json_path = project / f"{args.experiment_id}_summary.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        f"# SAR-Mix Proxy: {args.experiment_id}",
        "",
        "All checkpoints were selected on the same 400-image mixed validation split.",
        "",
        "## Overall Ranking",
        "",
        "| Rank | Experiment | Variant | Placement | P | R | AP50 | AP75 | AP50:95 |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, item in enumerate(overall, 1):
        lines.append(
            f"| {rank} | {item['experiment']} | {item['variant']} | {item['placement']} | "
            f"{item['precision']:.4f} | {item['recall']:.4f} | {item['ap50']:.4f} | "
            f"{item['ap75']:.4f} | {item['ap50_95']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Per-Source Validation",
            "",
            "| Experiment | Source | AP50 | AP75 | AP50:95 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for item in rows:
        if item["subset"] == "overall":
            continue
        lines.append(
            f"| {item['experiment']} | {item['subset']} | {item['ap50']:.4f} | "
            f"{item['ap75']:.4f} | {item['ap50_95']:.4f} |"
        )
    md_path = project / f"{args.experiment_id}_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"completed experiments: {len(overall)}")
    if overall:
        print(f"best overall: {overall[0]['experiment']} AP50:95={overall[0]['ap50_95']:.6f}")
    if best_placement:
        print(
            f"best full placement: {best_placement['placement']} "
            f"AP50:95={best_placement['ap50_95']:.6f}"
        )
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")


if __name__ == "__main__":
    main()
