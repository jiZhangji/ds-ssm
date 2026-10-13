#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root = Path(args.project)
    rows = []
    for path in sorted(root.rglob("result.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            rows.append({"status": "invalid", "path": str(path), "error": str(exc)})
    payload = {"project": str(root.resolve()), "records": rows, "record_count": len(rows)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = Path(args.output).with_suffix(".md")
    lines = ["# IG-DS-SSM/PFIM Cross-Task Results", "", "| Dataset | Task | Variant | Metrics |", "|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row.get('dataset','')} | {row.get('task','')} | {row.get('variant','')} | `{json.dumps(row.get('metric',{}), ensure_ascii=False)}` |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {Path(args.output)}")


if __name__ == "__main__":
    main()
