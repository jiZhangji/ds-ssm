#!/usr/bin/env python3
"""Aggregate matched OGSOD TEST runs into mean/std and paired deltas."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


METRICS = ("precision", "recall", "ap50", "ap75", "ap50_95")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--matrix-id", required=True)
    parser.add_argument("--expected-seeds", default="0 1 2")
    parser.add_argument("--expected-imgsz", type=int, default=1024)
    parser.add_argument("--expected-precision", default="fp32")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
    }


def validate_record(payload: dict, path: Path, args: argparse.Namespace) -> None:
    errors = []

    expected_run_name = (
        f"{args.matrix_id}_{payload.get('variant', '')}_seed{payload.get('seed', '')}"
    )
    if payload.get("run_name") != expected_run_name:
        errors.append(
            f"run_name={payload.get('run_name')!r} != {expected_run_name!r}"
        )

    if payload.get("split") != "test":
        errors.append(f"split={payload.get('split')!r} != 'test'")

    if payload.get("precision") != args.expected_precision:
        errors.append(
            f"precision={payload.get('precision')!r} != {args.expected_precision!r}"
        )

    if payload.get("imgsz") != args.expected_imgsz:
        errors.append(
            f"imgsz={payload.get('imgsz')!r} != {args.expected_imgsz}"
        )

    for field in ("weights_sha256", "data_yaml_sha256", "evaluator_sha256"):
        value = payload.get(field)
        if not isinstance(value, str) or len(value) != 64:
            errors.append(f"{field}={value!r} is not a 64-char SHA-256 digest")

    if "overall" not in payload or "per_class" not in payload:
        errors.append("missing overall or per_class payload")

    if errors:
        raise SystemExit(
            f"Refusing test_metrics.json with non-matching evidence: {path}\n"
            + "\n".join(f"  - {error}" for error in errors)
        )


def main() -> None:
    args = parse_args()
    project = Path(args.project).resolve()
    expected_seeds = sorted(int(x) for x in args.expected_seeds.split())
    records = []
    for path in sorted(project.glob(f"{args.matrix_id}_*_seed*/test_metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        validate_record(payload, path, args)
        record = {
            "variant": payload["variant"],
            "seed": int(payload["seed"]),
            "run_name": payload["run_name"],
            "path": str(path),
            "per_class": {
                item["class_name"]: {
                    "ap50": float(item["ap50"]),
                    "ap75": float(item["ap75"]),
                    "ap50_95": float(item["ap50_95"]),
                }
                for item in payload["per_class"]
            },
        }
        record.update({key: float(payload["overall"][key]) for key in METRICS})
        records.append(record)

    if not records:
        raise SystemExit(f"No test_metrics.json files found below {project}")

    seen = set()
    for record in records:
        key = (record["variant"], record["seed"])
        if key in seen:
            raise SystemExit(f"Duplicate TEST records for variant/seed={key}")
        seen.add(key)

    raw_csv = project / f"{args.matrix_id}_test_runs.csv"
    with raw_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    variants = sorted({record["variant"] for record in records})
    summary = {}
    for variant in variants:
        rows = [record for record in records if record["variant"] == variant]
        seeds = sorted(row["seed"] for row in rows)
        complete = seeds == expected_seeds
        if args.require_complete and not complete:
            missing = sorted(set(expected_seeds) - set(seeds))
            raise SystemExit(
                f"--require-complete: variant={variant} is missing seeds {missing}; "
                "refusing to emit a publication table from incomplete runs"
            )

        class_names = sorted({name for row in rows for name in row["per_class"]})
        per_class = {}
        for class_name in class_names:
            per_class[class_name] = {
                metric: summarize(
                    [
                        row["per_class"][class_name][metric]
                        for row in rows
                        if class_name in row["per_class"]
                    ]
                )
                for metric in ("ap50", "ap75", "ap50_95")
            }

        summary[variant] = {
            "seeds": seeds,
            "complete": complete,
            "metrics": {
                metric: summarize([row[metric] for row in rows]) for metric in METRICS
            },
            "per_class": per_class,
        }

    by_variant_seed = {
        (record["variant"], record["seed"]): record for record in records
    }
    paired = {}
    if "baseline" in variants:
        for variant in variants:
            if variant == "baseline":
                continue
            seeds = sorted(
                seed
                for seed in expected_seeds
                if ("baseline", seed) in by_variant_seed
                and (variant, seed) in by_variant_seed
            )
            paired[variant] = {
                "seeds": seeds,
                "metrics": {
                    metric: summarize(
                        [
                            by_variant_seed[(variant, seed)][metric]
                            - by_variant_seed[("baseline", seed)][metric]
                            for seed in seeds
                        ]
                    )
                    for metric in METRICS
                }
                if seeds
                else {},
            }

    output = {
        "matrix_id": args.matrix_id,
        "project": str(project),
        "expected_seeds": expected_seeds,
        "runs": records,
        "summary": summary,
        "paired_delta_vs_baseline": paired,
    }
    json_path = project / f"{args.matrix_id}_summary.json"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    summary_csv = project / f"{args.matrix_id}_summary.csv"
    fields = ["variant", "seeds", "complete"]
    for metric in METRICS:
        fields.extend((f"{metric}_mean", f"{metric}_std"))
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant in variants:
            item = summary[variant]
            row = {
                "variant": variant,
                "seeds": " ".join(str(x) for x in item["seeds"]),
                "complete": item["complete"],
            }
            for metric in METRICS:
                row[f"{metric}_mean"] = item["metrics"][metric]["mean"]
                row[f"{metric}_std"] = item["metrics"][metric]["std"]
            writer.writerow(row)

    md_path = project / f"{args.matrix_id}_summary.md"
    lines = [
        f"# {args.matrix_id}",
        "",
        "All values below are TEST percentages from validation-selected best checkpoints.",
        "",
        "| Variant | Seeds | Complete | P | R | AP50 | AP75 | AP50:95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in variants:
        item = summary[variant]
        formatted = []
        for metric in METRICS:
            stat = item["metrics"][metric]
            formatted.append(f"{100 * stat['mean']:.3f} +/- {100 * stat['std']:.3f}")
        lines.append(
            f"| {variant} | {' '.join(map(str, item['seeds']))} | {item['complete']} | "
            + " | ".join(formatted)
            + " |"
        )

    if paired:
        lines.extend(
            [
                "",
                "## Paired Delta vs Baseline",
                "",
                "Deltas are percentage points, paired by seed.",
                "",
                "| Variant | Seeds | dAP50 | dAP75 | dAP50:95 |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for variant, item in paired.items():
            if not item["metrics"]:
                continue
            values = []
            for metric in ("ap50", "ap75", "ap50_95"):
                stat = item["metrics"][metric]
                values.append(f"{100 * stat['mean']:+.3f} +/- {100 * stat['std']:.3f}")
            lines.append(
                f"| {variant} | {' '.join(map(str, item['seeds']))} | "
                + " | ".join(values)
                + " |"
            )

    lines.extend(
        [
            "",
            "## Per-Class AP (mean +/- std, TEST)",
            "",
            "| Variant | Class | AP50 | AP75 | AP50:95 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for variant in variants:
        for class_name, stats in summary[variant]["per_class"].items():
            values = []
            for metric in ("ap50", "ap75", "ap50_95"):
                stat = stats[metric]
                values.append(f"{100 * stat['mean']:.3f} +/- {100 * stat['std']:.3f}")
            lines.append(
                f"| {variant} | {class_name} | " + " | ".join(values) + " |"
            )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"runs:    {raw_csv}")
    print(f"summary: {summary_csv}")
    print(f"json:    {json_path}")
    print(f"report:  {md_path}")


if __name__ == "__main__":
    main()
