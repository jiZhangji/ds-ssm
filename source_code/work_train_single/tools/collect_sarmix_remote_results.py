#!/usr/bin/env python3
"""Collect small evaluation JSONs from remote shards and build one ranking table."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARIZE = ROOT / "tools" / "summarize_sarmix.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Remote shard as SSH_HOST:/absolute/project/path; repeat for each server",
    )
    parser.add_argument("--experiment-id", default="sarmix_scratch_d16_v1")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def fetch(source: str, experiment_id: str) -> list[dict]:
    if ":" not in source:
        raise ValueError(f"Invalid source: {source}")
    host, remote_root = source.split(":", 1)
    if not host or not remote_root.startswith("/"):
        raise ValueError(f"Expected SSH_HOST:/absolute/path, got: {source}")
    snippet = (
        "import json,sys; from pathlib import Path; "
        "root=Path(sys.argv[1]); exp=sys.argv[2]; "
        "items=[]; "
        "[items.append(json.loads(p.read_text(encoding='utf-8'))) "
        "for p in sorted(root.glob(exp + '_*/eval_val_*.json'))]; "
        "print(json.dumps(items))"
    )
    command = (
        f"python3 -c {shlex.quote(snippet)} "
        f"{shlex.quote(remote_root)} {shlex.quote(experiment_id)}"
    )
    result = subprocess.run(
        ["ssh", host, command],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(result.stdout)


def main() -> None:
    args = parse_args()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = []
    seen = set()
    for source in args.source:
        items = fetch(source, args.experiment_id)
        print(f"{source}: {len(items)} evaluation records")
        for item in items:
            key = (item["experiment"], item["subset"])
            if key in seen:
                raise RuntimeError(f"Duplicate evaluation record: {key}")
            seen.add(key)
            records.append(item)

    if not records:
        raise SystemExit("No completed evaluation records found yet")
    for item in records:
        run_dir = out / item["experiment"]
        run_dir.mkdir(parents=True, exist_ok=True)
        subset = item["subset"].lower().replace("-", "_")
        (run_dir / f"eval_val_{subset}.json").write_text(
            json.dumps(item, indent=2), encoding="utf-8"
        )

    subprocess.run(
        [
            sys.executable,
            str(SUMMARIZE),
            "--project",
            str(out),
            "--experiment-id",
            args.experiment_id,
        ],
        check=True,
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()
