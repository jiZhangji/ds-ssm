#!/usr/bin/env python3
"""Stage an exact frozen detection split from explicitly named local sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from manifest import atomic_json, frozen, sha, verify_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-manifest", required=True)
    p.add_argument("--source", action="append", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dataset", choices=("ssdd", "sadd"), required=True)
    args = p.parse_args()
    source_manifest = Path(args.source_manifest).resolve()
    sources = [Path(s).resolve() for s in args.source]
    output = Path(args.output).resolve()
    if not output.is_absolute() or output.name != args.dataset:
        raise ValueError("Use an explicit dataset-specific output")
    if any(s == output or s in output.parents or output in s.parents for s in sources):
        raise ValueError("Staging output must be independent of source roots")
    if (output / "dataset.json").exists():
        info = json.loads((output / "dataset.json").read_text())
        print(json.dumps({"status": "already_verified", "dataset": info, "audit": verify_dataset(info)}))
        return
    rows = json.loads(source_manifest.read_text())
    rows = [{**r, "name": r.get("name", r.get("image"))} for r in rows]
    for row in rows:
        if row["split"] not in {"train", "val", "test"} or Path(row["name"]).name != row["name"]:
            raise ValueError("Unsafe manifest row")
    wanted = {row["name"] for row in rows} | {Path(row["name"]).stem + ".txt" for row in rows}
    by_name = {}
    for source in sources:
        if not source.is_dir():
            continue
        for item in source.rglob("*"):
            if item.name in wanted and item.is_file():
                by_name.setdefault(item.name, []).append(item)
    copies = []
    counts = {}
    for row in rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
        for kind, name, key in (("images", row["name"], "image_sha256"),
                                ("labels", Path(row["name"]).stem + ".txt", "label_sha256")):
            destination = output / "data" / kind / row["split"] / name
            if destination.exists():
                if sha(destination) != row[key]:
                    raise RuntimeError(f"Existing staged file differs: {destination}")
                continue
            match = None
            data = None
            normalized = False
            for candidate in by_name.get(name, []):
                candidate_data = candidate.read_bytes()
                if hashlib.sha256(candidate_data).hexdigest() == row[key]:
                    match, data = candidate, candidate_data
                    break
                if kind == "labels":
                    fixed = candidate_data.replace(b"\r\n", b"\n")
                    if hashlib.sha256(fixed).hexdigest() == row[key]:
                        match, data, normalized = candidate, fixed, True
                        break
            if match is None:
                raise RuntimeError(f"No exact source bytes found for {kind}/{row['split']}/{name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as f:
                f.write(data)
            copies.append({"destination": str(destination), "source": str(match),
                           "sha256": row[key], "crlf_to_lf": normalized})
    import yaml
    yaml_path = output / "data.yaml"
    payload = yaml.safe_dump({"path": str(output / "data"), "train": "images/train", "val": "images/val",
                             "test": "images/test", "names": {0: "ship" if args.dataset == "ssdd" else "aircraft"}},
                            sort_keys=False)
    output.mkdir(parents=True, exist_ok=True)
    if yaml_path.exists() and yaml_path.read_text() != payload:
        raise RuntimeError("Existing data YAML differs")
    if not yaml_path.exists():
        with yaml_path.open("x") as f:
            f.write(payload)
    manifest_path = output / "data_content.json"
    # Preserve exact frozen bytes when no legacy row-name conversion is necessary.
    if all("name" in r for r in json.loads(source_manifest.read_text())):
        if manifest_path.exists() and sha(manifest_path) != sha(source_manifest):
            raise RuntimeError("Existing content manifest differs")
        if not manifest_path.exists():
            shutil.copyfile(source_manifest, manifest_path)
    else:
        frozen(manifest_path, rows)
    info = {"protocol": args.dataset, "root": str(output / "data"), "yaml": str(yaml_path),
            "yaml_sha256": sha(yaml_path), "manifest": str(manifest_path),
            "manifest_sha256": sha(manifest_path), "counts": counts}
    audit = verify_dataset(info)
    frozen(output / "dataset.json", info)
    frozen(output / "staging_audit.json", {"status": "verified", "source_manifest": str(source_manifest),
           "source_manifest_sha256": sha(source_manifest), "sources": [str(s) for s in sources],
           "copies": copies, "audit": audit,
           "limitations": "Exact-content split matching, not a scene-disjoint or near-duplicate proof."})
    print(json.dumps({"status": "verified", "dataset": info, "copied": len(copies)}))


if __name__ == "__main__":
    main()
