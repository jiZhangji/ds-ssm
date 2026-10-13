#!/usr/bin/env python3
"""Trusted, server-only coordinator for all 57 seed0 completion receipts.

Inputs are complete producing-host verify_receipts.py stdout snapshots, or an
already-issued candidate barrier. This does NOT independently recheck remote
artifacts, authenticate a host, or provide a cryptographic signature. No SSH,
GPU import, experiment execution, scheduler discovery or automatic retry.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import socket
import sys
import uuid
from pathlib import Path

import manifest
from verify_receipts import expected_artifacts

HOST_KEYS = ("a", "b", "c", "f0", "f1")
SEED0_COUNT = 57
SNAPSHOT_SCHEMA = "rgsp-verified-receipt-snapshot-v1"
COLLECTION_SCHEMA = "rgsp-trusted-global-collection-v1"
TRUST = ("Trusted coordinator accepting producing-host verification snapshots; "
         "remote artifact files are not independently rechecked here; hashes are not host authentication or a signature")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def exact(actual, expected, message):
    require(type(actual) is type(expected) and manifest.canonical(actual) == manifest.canonical(expected), message)


def hex_sha(value, message):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def read_document(path):
    path = manifest.absolute(path)
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object, parse_constant=reject_constant)
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def plan_contract(plan):
    """This helper deliberately cannot reduce or silently change the matrix."""
    require(not plan.get("smoke_only", False), "Smoke plans cannot publish this formal seed0 barrier")
    expected = sorted(plan["seed0_trial_ids"])
    require(len(expected) == SEED0_COUNT and len(set(expected)) == SEED0_COUNT,
            f"This coordinator requires all {SEED0_COUNT} distinct planned seed0 trials")
    require(isinstance(plan.get("dispatch_map"), dict) and set(plan["dispatch_map"]) == set(HOST_KEYS),
            "The plan must declare exactly a/b/c/f0/f1 dispatch entries")
    manifest.within(plan["barrier_path"], plan["campaign_root"])
    lanes = {lane["id"]: lane for lane in plan["lanes"]}
    selected, assigned = {}, []
    for key in HOST_KEYS:
        entry = plan["dispatch_map"][key]
        lane = lanes[entry["seed0_lane"]]
        exact(lane["hostname"], entry["hostname"], f"Host differs between dispatch and seed0 lane: {key}")
        exact(lane["gpu_uuid"], entry["gpu_uuid"], f"GPU differs between dispatch and seed0 lane: {key}")
        require(lane["trial_ids"], f"Required producing-host seed0 lane is empty: {key}")
        for tid in lane["trial_ids"]:
            trial = manifest.trial_by_id(plan, tid)
            exact(trial["seed"], 0, f"A non-seed0 trial appears in {key}: {tid}")
            require(trial["kind"] != "not_ready", f"Planned seed0 remains not_ready: {tid}")
        selected[key] = lane
        assigned.extend(lane["trial_ids"])
    require(len(assigned) == len(set(assigned)) and sorted(assigned) == expected,
            "The five producing-host lanes must cover the entire seed0 matrix exactly once")
    return expected, selected


def validate_completion(plan, tid, receipt):
    trial = manifest.trial_by_id(plan, tid)
    require(trial["kind"] != "not_ready", f"A not-ready trial cannot have accepted completion: {tid}")
    require(isinstance(receipt, dict), f"Receipt must be an object: {tid}")
    manifest.validate_receipt(plan, trial, receipt, verify_files=False)
    exact(receipt["seed"], 0, f"Receipt seed is not an integer zero: {tid}")
    require(isinstance(receipt["artifacts"], list), f"Receipt artifacts must be an ordered list: {tid}")
    exact([item["path"] for item in receipt["artifacts"]], expected_artifacts(trial),
          f"Receipt does not have the exact ordered detection/command/reuse artifacts: {tid}")
    require(not receipt.get("details", {}).get("smoke_only", False), f"Smoke-only receipt cannot complete formal trial: {tid}")
    return receipt


def validate_snapshots(plan, paths):
    expected, lanes = plan_contract(plan)
    require(len(paths) == len(HOST_KEYS), "Exactly five complete producing-host snapshots are required")
    seen, receipts, evidence, documents = set(), {}, [], {}
    for path in paths:
        snapshot, source = read_document(path)
        exact(snapshot.get("schema"), SNAPSHOT_SCHEMA, f"Unknown snapshot schema: {path}")
        exact(snapshot.get("plan_sha256"), manifest.digest(plan), f"Snapshot belongs to another immutable plan: {path}")
        key = snapshot.get("host_key")
        require(isinstance(key, str) and key in HOST_KEYS and key not in seen,
                f"Missing, unknown or duplicate producing host_key: {key!r}")
        seen.add(key)
        lane, entry = lanes[key], plan["dispatch_map"][key]
        tids = lane["trial_ids"]
        exact(snapshot.get("hostname"), entry["hostname"], f"Snapshot producing hostname differs: {key}")
        exact(snapshot.get("lane"), lane["id"], f"Snapshot selected lane differs: {key}")
        exact(snapshot.get("seed"), 0, f"Snapshot is not the seed0 verification: {key}")
        exact(snapshot.get("requested_trial_ids"), tids, f"Snapshot did not check the exact full assigned lane: {key}")
        exact(snapshot.get("all_requested_complete"), True, f"Snapshot still has incomplete requested trials: {key}")
        exact(snapshot.get("requested_count"), len(tids), f"Snapshot requested count differs: {key}")
        exact(snapshot.get("verified_count"), len(tids), f"Snapshot verified count differs: {key}")
        exact(snapshot.get("global_seed0_trial_ids"), expected, f"Snapshot carries a different global seed0 set: {key}")
        exact(snapshot.get("global_seed0_not_ready_ids"), [], f"Snapshot reports not-ready seed0 rows: {key}")
        remaining = sorted(set(expected) - set(tids))
        exact(snapshot.get("global_seed0_unverified_in_this_snapshot"), remaining,
              f"Snapshot silently changed its local/global verification scope: {key}")
        exact(snapshot.get("all_global_seed0_verified_locally"), not remaining,
              f"Snapshot incorrectly asserts global completion: {key}")
        require(isinstance(snapshot.get("checked_utc"), str) and snapshot["checked_utc"], f"Snapshot timestamp missing: {key}")
        rows, by_id = snapshot.get("rows"), snapshot.get("verified_receipts")
        require(isinstance(rows, list) and len(rows) == len(tids), f"Snapshot rows do not cover the full lane: {key}")
        require(isinstance(by_id, dict) and set(by_id) == set(tids), f"Snapshot receipt map does not cover the full lane: {key}")
        row_ids, receipt_hashes = [], {}
        for row in rows:
            require(isinstance(row, dict), f"Invalid row in snapshot: {key}")
            tid = row.get("trial_id")
            require(isinstance(tid, str) and tid in tids and tid not in row_ids,
                    f"Unknown or duplicate snapshot row: {key}, {tid!r}")
            row_ids.append(tid)
            trial = manifest.trial_by_id(plan, tid)
            exact(row.get("status"), "verified", f"Snapshot row is not verified: {tid}")
            exact(row.get("seed"), 0, f"Snapshot row is not seed0: {tid}")
            exact(row.get("kind"), trial["kind"], f"Snapshot row kind differs from the plan: {tid}")
            exact(row.get("receipt_path"), str(manifest.receipt_path(plan, trial)), f"Snapshot receipt path differs: {tid}")
            hex_sha(row.get("receipt_sha256"), f"Snapshot lacks a full producing-host receipt SHA256: {tid}")
            exact(row.get("receipt"), by_id[tid], f"Row and verified_receipts contain different receipts: {tid}")
            receipt = validate_completion(plan, tid, by_id[tid])
            exact(row.get("artifacts_verified"), len(receipt["artifacts"]), f"Snapshot artifact verification count differs: {tid}")
            require(tid not in receipts, f"A seed0 receipt appeared in multiple host snapshots: {tid}")
            receipts[tid] = receipt
            receipt_hashes[tid] = row["receipt_sha256"]
        require(set(row_ids) == set(tids), f"Snapshot omitted requested rows: {key}")
        source.update(host_key=key, hostname=entry["hostname"], lane=lane["id"], checked_utc=snapshot["checked_utc"],
                      producing_host_receipt_sha256=receipt_hashes)
        evidence.append(source)
        documents[key] = snapshot
    require(seen == set(HOST_KEYS) and sorted(receipts) == expected,
            "All five producing hosts and all 57 seed0 receipts are required; no partial barrier")
    return receipts, sorted(evidence, key=lambda item: item["host_key"]), documents


def validate_candidate(plan, marker):
    expected, unused_lanes = plan_contract(plan)
    require(isinstance(marker, dict) and set(marker) == {"payload", "payload_sha256"}, "Invalid barrier envelope")
    payload = marker["payload"]
    require(isinstance(payload, dict), "Barrier payload must be an object")
    hex_sha(marker["payload_sha256"], "Barrier payload SHA256 is missing or malformed")
    exact(marker["payload_sha256"], manifest.digest(payload), "Candidate barrier payload checksum differs")
    exact(payload.get("schema"), manifest.SCHEMA, "Candidate barrier schema differs")
    exact(payload.get("plan_sha256"), manifest.digest(plan), "Candidate barrier belongs to a different immutable plan")
    exact(payload.get("state"), "all_seed0_complete", "Candidate barrier does not assert all seed0 complete")
    exact(payload.get("seed0_trial_ids"), expected, "Candidate barrier does not cover exactly all 57 seed0 trials")
    receipts = payload.get("receipts")
    require(isinstance(receipts, dict) and sorted(receipts) == expected, "Candidate barrier omitted or added seed0 receipts")
    for tid in expected:
        validate_completion(plan, tid, receipts[tid])
    require(isinstance(payload.get("issued_utc"), str) and payload["issued_utc"], "Candidate barrier lacks issue timestamp")
    require(isinstance(payload.get("trust_model"), str) and payload["trust_model"], "Candidate barrier lacks declared trust model")
    return payload


@contextlib.contextmanager
def publication_lock(plan):
    import fcntl
    path = manifest.within(str(manifest.absolute(plan["campaign_root"]) / "locks" / "trusted_seed0_coordinator.lock"),
                           plan["campaign_root"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another local trusted coordinator holds the publication lock; no retry")
        yield


def collect(plan, paths, collected_dir=None, validate_only=False):
    receipts, evidence, documents = validate_snapshots(plan, paths)
    result = {"schema": COLLECTION_SCHEMA, "mode": "collect_snapshots", "plan_sha256": manifest.digest(plan),
              "seed0_count": len(receipts), "host_keys": list(HOST_KEYS), "source_snapshots": evidence,
              "trust_model": TRUST, "remote_artifacts_rechecked_here": False}
    if validate_only:
        return {**result, "status": "validated_only", "writes_performed": False}
    with publication_lock(plan):
        barrier_path = manifest.absolute(plan["barrier_path"])
        if barrier_path.exists():
            existing, unused_source = read_document(barrier_path)
            payload = validate_candidate(plan, existing)
            exact(payload["receipts"], receipts, "An existing barrier has different receipt evidence; no overwrite")
            manifest.verify_barrier(plan)
            return {**result, "status": "already_published", "barrier_path": str(barrier_path),
                    "barrier_sha256": manifest.sha(barrier_path), "new_collection_created": False}
        evidence_root = manifest.within(str(manifest.absolute(plan["campaign_root"]) / "evidence"), plan["campaign_root"])
        if collected_dir is None:
            tag = manifest.utc().replace(":", "").replace("+", "_") + "_" + uuid.uuid4().hex[:8]
            collected_dir = str(evidence_root / ("collected_seed0_" + tag))
        target = manifest.within(collected_dir, str(evidence_root))
        require(not target.exists(), "Collection directory must be new; preserve previous collection evidence")
        # Nothing has been collected until every snapshot has passed all checks.
        target.mkdir(parents=True, exist_ok=False)
        for tid in sorted(receipts):
            manifest.frozen(target / (tid + ".json"), receipts[tid])
        for key in HOST_KEYS:
            manifest.frozen(target / "source_snapshots" / (key + ".json"), documents[key])
        manifest.frozen(target / "collection_verified.json", {**result, "status": "all_seed0_snapshots_verified",
                        "verified_utc": manifest.utc(), "collector_sha256": manifest.sha(__file__)})
        manifest.issue_barrier(plan, str(target))
        marker, unused_source = read_document(barrier_path)
        payload = validate_candidate(plan, marker)
        exact(payload["receipts"], receipts, "Published barrier receipt content differs from the collected evidence")
        manifest.verify_barrier(plan)
        completed = {**result, "status": "published", "collected_dir": str(target),
                     "barrier_path": str(barrier_path), "barrier_sha256": manifest.sha(barrier_path),
                     "payload_sha256": marker["payload_sha256"], "published_utc": manifest.utc()}
        manifest.frozen(target / "barrier_issue_complete.json", completed)
        return completed


def accept(plan, candidate, validate_only=False):
    marker, source = read_document(candidate)
    payload = validate_candidate(plan, marker)
    result = {"schema": COLLECTION_SCHEMA, "mode": "accept_barrier", "plan_sha256": manifest.digest(plan),
              "seed0_count": len(payload["receipts"]), "candidate": source,
              "payload_sha256": marker["payload_sha256"], "trust_model": TRUST,
              "remote_artifacts_rechecked_here": False}
    if validate_only:
        return {**result, "status": "validated_only", "writes_performed": False}
    with publication_lock(plan):
        path = manifest.absolute(plan["barrier_path"])
        existed = path.exists()
        manifest.frozen(path, marker)  # Different bytes fail closed; nothing is replaced.
        manifest.verify_barrier(plan)
        installed, unused_source = read_document(path)
        validate_candidate(plan, installed)
        exact(installed, marker, "Installed barrier differs from validated candidate")
        return {**result, "status": "already_published" if existed else "accepted",
                "barrier_path": str(path), "barrier_sha256": manifest.sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--expected-plan-sha256", help="Optional independent pin of the canonical immutable plan SHA256")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--snapshot", action="append", help="Complete producing-host stdout JSON; repeat exactly five times")
    mode.add_argument("--accept-barrier", help="Explicit candidate file; validate then freeze at this plan's barrier_path")
    parser.add_argument("--collected-dir", help="New directory strictly inside campaign_root/evidence; default is a unique new directory")
    parser.add_argument("--validate-only", action="store_true", help="Read-only validation; no directory, lock, receipt or barrier writes")
    args = parser.parse_args()
    require(sys.platform == "linux", "Server Linux only; no workstation execution")
    if args.accept_barrier and args.collected_dir:
        parser.error("--collected-dir applies only to snapshot collection")
    raw_plan, plan_file = read_document(args.plan)
    plan = manifest.load_plan(args.plan)
    exact(plan, raw_plan, "Plan changed while loading")
    plan_contract(plan)
    require(socket.gethostname() in {entry["hostname"] for entry in plan["dispatch_map"].values()},
            "Coordinator/acceptance must run on an explicitly declared campaign host")
    if args.expected_plan_sha256:
        hex_sha(args.expected_plan_sha256, "Expected plan SHA256 must be 64 lowercase hex characters")
        exact(manifest.digest(plan), args.expected_plan_sha256, "The explicit expected immutable plan hash differs")
    manifest.check_file(plan_file)
    if args.snapshot:
        result = collect(plan, args.snapshot, args.collected_dir, args.validate_only)
    else:
        result = accept(plan, args.accept_barrier, args.validate_only)
    result.update(coordinator_hostname=socket.gethostname(), plan_file=plan_file, finished_utc=manifest.utc())
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
