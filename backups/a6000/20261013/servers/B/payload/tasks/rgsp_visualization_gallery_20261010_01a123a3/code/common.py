"""Shared, deterministic styling and labels for the RGSP figure collection.

This package reads recorded results only. It never imports a detector, loads
checkpoints, starts training, or performs inference.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = {
    "ink": "#283846", "muted": "#687787", "grid": "#DFE5EA",
    "blue": "#477DA7", "teal": "#358D87", "orange": "#C58B4D",
    "red": "#B95C63", "purple": "#8A75AA", "gray": "#88939E",
    "lightblue": "#EAF1F8", "lightteal": "#E9F4F1",
    "lightorange": "#FAF0E4", "lightred": "#F7EAEC",
}
COLORS = [C[k] for k in ("gray", "blue", "orange", "teal", "purple", "red")]
LABELS = {
    "original": "Original", "baseline": "Original",
    "matched_identity": "Matched identity", "matched_scale075": "Matched075",
    "local_energy": "LocalEnergy (R+E)", "full": "Full (P3)",
    "full_no_lrc": "Full no-LRC", "const_guidance": "Constant guidance",
    "post_gate": "Post-gate", "local_energy_input": "LocalEnergy + I-to-U",
    "local_energy_delta": "LocalEnergy + I-to-delta",
    "full_fixed_half": "Full fixed half", "residual_scale075": "Residual scale075 (R)",
    "energy_only": "Energy only (E)", "guidance_only": "Guidance only (I)",
    "global": "Global", "lsgm_trend_only": "LSGM trend only",
    "lsgm_std_only": "LSGM std only", "lsgm_point_only": "LSGM point only",
    "full_p2": "Full (P2)", "full_p4": "Full (P4)",
    "cnn_yolov8n": "CNN YOLOv8n", "full_d8": "Full d8", "full_d32": "Full d32",
    "full_trend5": "Full trend5", "full_stats5": "Full stats5",
    "full_trend7": "Full trend7", "full_stats7": "Full stats7",
    "full_p3": "Full (legacy)", "ds_only_p3": "DS-only (legacy)",
    "pfim_only_p3": "PFIM-only* (legacy)", "global_p3": "Global (legacy)",
}
METRICS = {"ap50": "AP50 (%)", "ap75": "AP75 (%)", "ap50_95": "AP50:95 (%)",
           "precision": "Precision (%)", "recall": "Recall (%)"}
NEW_SOURCE = "inputs/latest_seed0.json"
NEW_NOTE = "Recorded test results | seed 0 only | snapshot: 2026-10-10 10:07 CST"
LEGACY_NOTE = ("Historical SSDD | 742/186/232 | 3 split/training seeds\n"
               "Within-seed paired; sample SD includes split and training variation. Separate from cleaned split.")
SADD_NOTE = "SADD historical 12-run campaign | n = 3 | positive-only test, 44 images"


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "text.color": C["ink"], "axes.labelcolor": C["ink"],
        "xtick.color": C["ink"], "ytick.color": C["ink"],
        "axes.edgecolor": "#A7B0BA", "axes.linewidth": .7,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": C["grid"], "grid.linewidth": .6,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "rgsp-gallery-20261010", "legend.frameon": False,
    })


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def current_records(data_dir):
    source = read_json(Path(data_dir) / "latest_seed0.json")
    records = source["records"]
    ids = [r["trial_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate recorded trial IDs")
    for r in records:
        if r["seed"] != 0 or r["split"] != "test":
            raise ValueError("Expected seed0/test only")
        for key in METRICS:
            raw = float(r["metrics_raw"][key])
            value = float(r["metrics_percent"][key])
            if not np.isfinite(value) or not 0 <= value <= 100:
                raise ValueError(f"Invalid metric: {r['trial_id']} / {key}")
            if not np.isclose(raw * 100, value, atol=1e-9, rtol=0):
                raise ValueError("Metric scale mismatch")
        r["key"] = r["trial_id"].split("_", 1)[1].removesuffix("_seed0")
    return source, records


def label(key):
    return LABELS.get(key, key.replace("_", " "))


def color(key):
    return {"original": C["gray"], "baseline": C["gray"],
            "matched_scale075": C["blue"], "matched_identity": "#A5B5C5",
            "full": C["teal"], "full_p3": C["teal"],
            "post_gate": C["orange"], "local_energy": C["purple"],
            "ds_only_p3": C["orange"], "pfim_only_p3": C["purple"],
            "global_p3": C["red"]}.get(key, C["blue"])


def finish(fig, title, note, bottom=.16, top=.85):
    fig.suptitle(title, x=.055, y=.98, ha="left", va="top", fontsize=14,
                 fontweight="medium")
    fig.text(.055, .02, note, fontsize=8.5, color=C["muted"], va="bottom")
    fig.subplots_adjust(left=.14, right=.96, bottom=bottom, top=top,
                        wspace=.36, hspace=.45)


def grid(ax, axis="x"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, alpha=.8)


def metric_value(record, metric="ap50_95"):
    return float(record["metrics_percent"][metric])


def short_source(*paths):
    return list(paths)
