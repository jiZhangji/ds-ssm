#!/usr/bin/env python3
"""Read-only, complete JSON snapshot of locally verified campaign receipts.

No GPU allocation, external writes, SSH or barrier creation. Missing/not-ready
trials remain explicit. A verified local subset never means global completion.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

from manifest import (absolute, detection_artifacts, digest, is_detection_trial,
                      load_plan, read_json, receipt_path, sha, trial_by_id, utc,
                      validate_receipt, verify_code)


def expected_artifacts(trial):
    if is_detection_trial(trial):
        return detection_artifacts(trial)
    if trial["kind"] == "command":
        return trial["success_artifacts"]
    if trial["kind"] == "reuse":
        return [trial["reuse"][key]["path"] for key in
                ("train_complete", "eval_test", "evaluation_complete", "best", "protocol", "audit")]
    raise ValueError(f"No completion contract for trial kind: {trial['kind']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--host-key", required=True, choices=("a", "b", "c", "f0", "f1"))
    parser.add_argument("--seed", type=int, choices=(0, 1), default=0)
    parser.add_argument("--require-complete", action="store_true", help="Exit 2 if this full requested subset is incomplete")
    args = parser.parse_args()
    plan = load_plan(args.plan)
    entry = plan["dispatch_map"][args.host_key]
    if socket.gethostname() != entry["hostname"]:
        raise RuntimeError("Receipt checks must run on the producing host")
    verify_code(plan)
    lane = next(item for item in plan["lanes"] if item["id"] == entry[f"seed{args.seed}_lane"])
    if lane["hostname"] != entry["hostname"] or lane["gpu_uuid"] != entry["gpu_uuid"]:
        raise RuntimeError("Receipt lane differs from dispatch host/GPU assignment")
    selected = list(lane["trial_ids"])
    if len(selected) != len(set(selected)) or any(trial_by_id(plan, tid)["seed"] != args.seed for tid in selected):
        raise RuntimeError("Requested receipt subset is not exactly the selected seed lane")
    rows, verified = [], {}
    for tid in selected:
        trial = trial_by_id(plan, tid)
        path = receipt_path(plan, trial)
        row = {"trial_id": tid, "seed": trial["seed"], "kind": trial["kind"], "receipt_path": str(path)}
        if trial["kind"] == "not_ready":
            row.update(status="not_ready", reason="Planned but not implemented; never treated as completed")
        elif not path.exists():
            row.update(status="missing")
        else:
            try:
                receipt = validate_receipt(plan, trial, read_json(path))
                if [item["path"] for item in receipt["artifacts"]] != expected_artifacts(trial):
                    raise RuntimeError("Receipt does not contain the exact ordered semantic completion artifacts")
                validate_receipt(plan, trial, receipt, verify_files=True)
                if not plan.get("smoke_only", False) and receipt.get("details", {}).get("smoke_only", False):
                    raise RuntimeError("Smoke-only receipt cannot satisfy a formal trial")
                verified[tid] = receipt
                row.update(status="verified", receipt_sha256=sha(path), artifacts_verified=len(receipt["artifacts"]),
                           receipt=receipt)
            except Exception as error:
                row.update(status="invalid", error=f"{type(error).__name__}: {error}")
        rows.append(row)
    seed0_all = sorted(plan["seed0_trial_ids"])
    seed0_not_ready = sorted(tid for tid in seed0_all if trial_by_id(plan, tid)["kind"] == "not_ready")
    seed0_here = sorted(tid for tid in verified if trial_by_id(plan, tid)["seed"] == 0)
    remaining_global = sorted(set(seed0_all) - set(seed0_here))
    all_subset = all(row["status"] == "verified" for row in rows)
    report = {"schema": "rgsp-verified-receipt-snapshot-v1", "plan_sha256": digest(plan),
              "hostname": socket.gethostname(), "host_key": args.host_key, "seed": args.seed,
              "checked_utc": utc(), "lane": lane["id"], "requested_trial_ids": selected,
              "all_requested_complete": all_subset, "verified_count": len(verified), "requested_count": len(selected),
              "rows": rows, "verified_receipts": verified, "global_seed0_trial_ids": seed0_all,
              "global_seed0_not_ready_ids": seed0_not_ready, "global_seed0_unverified_in_this_snapshot": remaining_global,
              "all_global_seed0_verified_locally": not remaining_global and not seed0_not_ready,
              "scope": "Read-only verification of the full explicit local lane subset; combine producing-host snapshots before a trusted global barrier"}
    # Complete untruncated stdout is the collection interface. The caller should
    # retain the entire JSON in its evidence store, not only a display excerpt.
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    if args.require_complete and not all_subset:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
