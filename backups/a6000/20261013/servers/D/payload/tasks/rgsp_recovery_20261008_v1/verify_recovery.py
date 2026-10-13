#!/usr/bin/env python3
"""CPU-only final audit; reconstruct derived CSV from saved checkpoint records."""
import argparse
import csv
import io
import json
import os
import sys
import time
from pathlib import Path
import recovery_queue as r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--key", choices=r.HOSTS, required=True)
    a = p.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    cfg = r.layout(a.key)
    out = Path(cfg["output"])
    cfg = r.read(out / "recovery_manifest.json")
    if (out / "verification.json").exists():
        raise RuntimeError("Already finalized; inspect the existing audit")
    sys.path[:0] = [cfg["code"], str(Path(cfg["code"]) / "tools")]
    import torch
    torch.set_num_threads(1)
    cp = cfg["checkpoint"]
    run = out / "runs" / cp["variant"]
    assert r.sha(run / "weights/last.pt") == cp["sha256"]
    ckpt = torch.load(run / "weights/last.pt", map_location="cpu")
    tensors = 0

    def finite(obj):
        nonlocal tensors
        if torch.is_tensor(obj):
            assert torch.isfinite(obj).all(), "Nonfinite optimizer tensor"
            tensors += 1
        elif isinstance(obj, dict):
            for value in obj.values():
                finite(value)
        elif isinstance(obj, (tuple, list)):
            for value in obj:
                finite(value)

    finite(ckpt["optimizer"])
    records = ckpt["train_results"]
    assert records["epoch"][-1] == cp["completed_epoch"]
    path = run / "results.csv"
    before = r.sha(path)
    old_rows = list(csv.DictReader(path.open()))
    names = [x.strip() for x in old_rows[0]]
    assert names == list(records)
    assert len({len(values) for values in records.values()}) == 1
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(names)
    writer.writerows(zip(*(records[name] for name in names)))
    path.write_text(buffer.getvalue())
    controls = [json.loads(line) for line in (run / "controls_by_epoch.jsonl").read_text().splitlines()]
    # Do not fabricate any missing control audit rows from the EMA tensors.
    missing = sorted(set(range(1, cp["completed_epoch"] + 1)) - {row["epoch"] for row in controls})
    for name, digest in cfg["original_artifacts"].items():
        assert r.sha(Path(cfg["source_lane"]) / name) == digest, name
    protocol = r.code_audit(cfg)
    report = {"status": "PASS", "key": a.key, "checked_unix": time.time(),
              "original_artifacts_unchanged": len(cfg["original_artifacts"]),
              "frozen_code_files_verified": len(protocol["code"]),
              "optimizer_tensors_finite": tensors, "checkpoint_epoch": cp["completed_epoch"],
              "csv_source": "checkpoint.train_results", "previous_csv_rows": len(old_rows),
              "restored_csv_rows": len(records["epoch"]), "csv_before_sha256": before,
              "csv_after_sha256": r.sha(path), "missing_historical_control_epochs": missing,
              "missing_audit_policy": "No synthetic values; retain explicit gap, continue future real audits",
              "gpu_training_executed": False, "verifier_sha256": r.sha(Path(__file__))}
    cfg["csv_reconstruction"] = report
    r.write(out / "recovery_manifest.json", cfg)
    r.write(out / "verification.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
