#!/usr/bin/env python3
"""Aggregate SSDD multi-seed results across experiment-ids into RESULTS.md + results.csv.

Reads all `{experiment_id}_*/eval_test.json` under a project dir, groups by variant,
computes per-variant mean卤std and paired deltas versus baseline (within-seed pairing).

Usage (on remote):
  python aggregate_results.py --project ~/ig-ds-ssm/experiments/ssdd_paper_v1 \
      --experiment-ids seed0 seed1 seed2 --out-dir /tmp/aggregate

Fairness notes:
  - Pairing is within-seed (same split), so deltas are computed per seed then averaged.
  - Only seeds present in BOTH the variant and baseline are used for that delta.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

VARIANTS = ("baseline", "full_p3", "ds_only_p3", "pfim_only_p3", "global_p3")
METRICS = ("ap50", "ap75", "ap50_95", "precision", "recall")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--experiment-ids", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    project = Path(args.project).expanduser().resolve()
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # variant -> seed -> metrics
    data: dict[str, dict[str, dict]] = {v: {} for v in VARIANTS}

    for eid in args.experiment_ids:
        for jp in sorted(project.glob(f"{eid}_*/eval_test.json")):
            item = json.loads(jp.read_text(encoding="utf-8"))
            variant = item.get("variant", "")
            seed = str(item.get("seed", "?"))
            # normalize variant key (eval_test stores 'full'/'ds_only'/'pfim_only'/'global'/'baseline')
            norm = variant
            if variant in ("full", "ds_only", "pfim_only", "global"):
                norm = f"{variant}_p3"
            if norm not in data:
                data[norm] = {}
            data[norm][seed] = item

    # Per-variant mean卤std
    rows = []
    for variant in VARIANTS:
        seeds = sorted(data[variant].keys())
        if not seeds:
            continue
        agg = {"variant": variant, "n_seeds": len(seeds), "seeds": ",".join(seeds)}
        for m in METRICS:
            vals = [data[variant][s]["overall"][m] for s in seeds]
            agg[f"{m}_mean"] = statistics.fmean(vals)
            agg[f"{m}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        # speed (inference ms) mean
        inf = [data[variant][s].get("speed_ms", {}).get("inference", 0.0) for s in seeds]
        agg["inference_ms_mean"] = statistics.fmean(inf) if inf else 0.0
        rows.append(agg)

    # Paired deltas vs baseline (within-seed)
    deltas = []
    base = data.get("baseline", {})
    for variant in VARIANTS:
        if variant == "baseline":
            continue
        common = sorted(set(data[variant].keys()) & set(base.keys()))
        if not common:
            continue
        for m in ("ap50", "ap75", "ap50_95"):
            per_seed = [data[variant][s]["overall"][m] - base[s]["overall"][m] for s in common]
            deltas.append(
                {
                    "variant": variant,
                    "metric": m,
                    "n_pairs": len(common),
                    "delta_mean": statistics.fmean(per_seed),
                    "delta_std": statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0,
                    "per_seed": {s: round(d, 4) for s, d in zip(common, per_seed)},
                }
            )

    # Write results.csv
    csv_path = out / "results.csv"
    fieldnames = ["variant", "n_seeds", "seeds",
                  "ap50_mean", "ap50_std", "ap75_mean", "ap75_std",
                  "ap50_95_mean", "ap50_95_std",
                  "precision_mean", "precision_std", "recall_mean", "recall_std",
                  "inference_ms_mean"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    # Write per-seed individual table
    indiv_csv = out / "results_individual.csv"
    with indiv_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "seed", "ap50", "ap75", "ap50_95", "precision", "recall",
                         "inference_ms", "weights"])
        for variant in VARIANTS:
            for seed in sorted(data[variant].keys()):
                it = data[variant][seed]
                writer.writerow([
                    variant, seed,
                    round(it["overall"]["ap50"], 6), round(it["overall"]["ap75"], 6),
                    round(it["overall"]["ap50_95"], 6),
                    round(it["overall"]["precision"], 6), round(it["overall"]["recall"], 6),
                    it.get("speed_ms", {}).get("inference", ""), it.get("weights_sha256", ""),
                ])

    # Write markdown
    md = ["# SSDD Multi-Seed Results", "",
          "> Generated by aggregate_results.py. All runs use scratch init, same split per seed,"
          " validation-selected checkpoint, official test split evaluation.",
          ""]
    md.append("## Per-variant summary (mean 卤 std)")
    md.append("")
    md.append("| Variant | n | AP50 | AP75 | AP50:95 | P | R | inf(ms) |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        def f(m):
            return f"{r[f'{m}_mean']:.4f}卤{r[f'{m}_std']:.4f}"
        md.append(f"| {r['variant']} | {r['n_seeds']} | {f('ap50')} | {f('ap75')} | "
                  f"{f('ap50_95')} | {f('precision')} | {f('recall')} | "
                  f"{r['inference_ms_mean']:.2f} |")
    md.append("")
    md.append("## Paired deltas vs baseline (within-seed)")
    md.append("")
    md.append("| Variant | Metric | 螖 mean | 螖 std | per-seed |")
    md.append("|---|---|---:|---:|---|")
    for d in deltas:
        md.append(f"| {d['variant']} | {d['metric']} | {d['delta_mean']:+.4f} | "
                  f"{d['delta_std']:.4f} | {d['per_seed']} |")
    md.append("")
    md.append("## Scientific questions")
    md.append("")
    md.append("- **Q1 (DS-SSM gain)**: `ds_only_p3 鈭?baseline`")
    md.append("- **Q2 (PFIM alone)**: `pfim_only_p3 鈭?baseline`")
    md.append("- **Q2 (PFIM on DS-SSM)**: `full_p3 鈭?ds_only_p3`")
    md.append("- **Q3 (token vs image importance)**: `full_p3 鈭?global_p3`")
    md.append("")
    (out / "RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {indiv_csv}")
    print(f"Wrote {out / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
