#!/usr/bin/env python3
"""Server-only, standard-library-only synthetic coordinator contract checks.

Every receipt, snapshot, dataset path and barrier below is in a fresh synthetic
fixture campaign under this probe's output directory. No real artifacts, real
plan, real barrier, GPU, training process, network or runtime source is changed.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path


def require(value, message):
    if not value:
        raise AssertionError(message)


def rejection(name, function, match=None):
    try:
        function()
    except (RuntimeError, ValueError) as error:
        if match is not None:
            require(match in str(error), f"Unexpected rejection for {name}: {error}")
        return {"case": name, "status": "PASS", "error": f"{type(error).__name__}: {error}"}
    raise AssertionError(f"Invalid coordinator input was accepted: {name}")


def make_fixture(campaign, coordinator_path, manifest, coordinator):
    campaign.mkdir(parents=True, exist_ok=False)
    trials, lane_tids = [], {key: [] for key in coordinator.HOST_KEYS}
    for index in range(57):
        tid = f"synthetic_seed0_{index:03d}"
        run = campaign / "synthetic_runs" / tid
        trial = {"id": tid, "seed": 0, "kind": "command", "cwd": str(campaign),
                 "command": [sys.executable, "-c", "pass"], "run_dir": str(run),
                 "success_artifacts": [str(run / "synthetic_result.json")]}
        if index == 0:
            trial = {"id": tid, "seed": 0, "kind": "detection", "run_dir": str(run),
                     "dataset": "synthetic", "config": str(campaign / "synthetic_config.yaml"),
                     "spec": {"control": "original", "placement": "baseline"}}
        elif index == 1:
            trial["success_artifacts"].append(str(run / "synthetic_second_result.json"))
        elif index == 2:
            trial = {"id": tid, "seed": 0, "kind": "reuse", "run_dir": str(run), "reuse": {
                key: {"path": str(campaign / "synthetic_origin" / (key + ".json")), "sha256": "c" * 64}
                for key in ("train_complete", "eval_test", "evaluation_complete", "best", "protocol", "audit")}}
        trials.append(trial)
        lane_tids[coordinator.HOST_KEYS[index % 5]].append(tid)
    lanes, dispatch = [], {}
    for key in coordinator.HOST_KEYS:
        hostname = socket.gethostname() if key == "a" else ("synthetic-host-f" if key.startswith("f") else "synthetic-host-" + key)
        gpu = "GPU-synthetic-" + key
        dispatch[key] = {"hostname": hostname, "gpu_uuid": gpu, "seed0_lane": key + "_seed0", "seed1_lane": key + "_seed1"}
        for seed in (0, 1):
            lanes.append({"id": key + f"_seed{seed}", "hostname": hostname, "gpu_uuid": gpu,
                          "output_dir": str(campaign / "synthetic_lanes" / (key + f"_seed{seed}")),
                          "trial_ids": lane_tids[key] if seed == 0 else []})
    plan = {"schema": manifest.SCHEMA, "campaign_root": str(campaign), "source_root": str(campaign / "unused_code"),
            "python": sys.executable, "barrier_path": str(campaign / "synthetic_seed0_barrier.json"),
            "receipts_dir": str(campaign / "synthetic_producer_receipts"), "trials": trials, "lanes": lanes,
            "dispatch_map": dispatch, "seed0_trial_ids": [trial["id"] for trial in trials],
            "datasets": {"synthetic": {"not_loaded_by_this_probe": True}},
            "code_hashes": [{"path": str(coordinator_path), "sha256": manifest.sha(coordinator_path)}]}
    plan_path = campaign / "synthetic_plan.json"
    manifest.frozen(plan_path, plan)
    manifest.load_plan(str(plan_path))
    receipts = {}
    for trial in trials:
        receipt = {"schema": manifest.SCHEMA, "status": "complete", "plan_sha256": manifest.digest(plan),
                   "trial_id": trial["id"], "trial_sha256": manifest.digest(trial), "seed": 0,
                   "finished_utc": "2026-10-09T00:00:00+00:00", "artifacts": [
                       {"path": path, "sha256": hashlib.sha256(path.encode()).hexdigest()}
                       for path in coordinator.expected_artifacts(trial)],
                   "details": {"synthetic_fixture_only": True}}
        receipts[trial["id"]] = receipt
        manifest.frozen(manifest.receipt_path(plan, trial), receipt)
    snapshots, paths = {}, []
    for key in coordinator.HOST_KEYS:
        tids = lane_tids[key]
        rows = []
        for tid in tids:
            trial = manifest.trial_by_id(plan, tid)
            receipt_path = manifest.receipt_path(plan, trial)
            rows.append({"trial_id": tid, "seed": 0, "kind": trial["kind"], "status": "verified",
                         "receipt_path": str(receipt_path), "receipt_sha256": manifest.sha(receipt_path),
                         "artifacts_verified": len(receipts[tid]["artifacts"]), "receipt": receipts[tid]})
        snapshot = {"schema": coordinator.SNAPSHOT_SCHEMA, "plan_sha256": manifest.digest(plan),
                    "hostname": dispatch[key]["hostname"], "host_key": key, "seed": 0, "checked_utc": manifest.utc(),
                    "lane": key + "_seed0", "requested_trial_ids": tids, "all_requested_complete": True,
                    "verified_count": len(tids), "requested_count": len(tids), "rows": rows,
                    "verified_receipts": {tid: receipts[tid] for tid in tids},
                    "global_seed0_trial_ids": sorted(plan["seed0_trial_ids"]), "global_seed0_not_ready_ids": [],
                    "global_seed0_unverified_in_this_snapshot": sorted(set(plan["seed0_trial_ids"]) - set(tids)),
                    "all_global_seed0_verified_locally": False, "scope": "synthetic standard-library fixture only"}
        path = campaign / "synthetic_snapshots" / (key + ".json")
        manifest.frozen(path, snapshot)
        snapshots[key] = snapshot
        paths.append(str(path))
    return plan, str(plan_path), snapshots, paths, receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hostname", required=True)
    args = parser.parse_args()
    require(sys.platform == "linux" and socket.gethostname() == args.hostname, "Only the explicitly selected server may run this probe")
    runtime, output = Path(args.runtime_root).resolve(), Path(args.output_dir).resolve()
    require(runtime in output.parents and output != runtime, "Fixture output must stay inside the independent task evidence")
    require(not (output / "coordinator_probe_report.json").exists(), "Do not overwrite prior probe evidence")
    sys.path.insert(0, str(runtime))
    import manifest
    script = Path(__file__).with_name("collect_global.py")
    ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    module_spec = importlib.util.spec_from_file_location("independent_probe_coordinator", script)
    coordinator = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(coordinator)
    require("torch" not in sys.modules, "Coordinator control imports touched torch")
    report = {"schema": "rgsp-coordinator-probe-v1", "hostname": socket.gethostname(), "python": platform.python_version(),
              "started_utc": manifest.utc(), "coordinator_sha256": manifest.sha(script),
              "probe_sha256": manifest.sha(__file__), "runtime_manifest_sha256": manifest.sha(runtime / "manifest.py"),
              "runtime_verifier_sha256": manifest.sha(runtime / "verify_receipts.py"),
              "scope": "57 synthetic receipts in each isolated fixture; no remote artifact verification, real plan/barrier or GPU"}
    cases = []

    def cli(plan_path, paths=None, candidate=None, pin=None):
        command = [sys.executable, "-B", str(script), "--plan", plan_path, "--validate-only"]
        if paths:
            for path in paths:
                command.extend(["--snapshot", path])
        else:
            command.extend(["--accept-barrier", str(candidate)])
        if pin is not None:
            command.extend(["--expected-plan-sha256", pin])
        env = os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(runtime))
        result = subprocess.run(command, env=env, cwd=output, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr.strip().splitlines()[-1])
        return json.loads(result.stdout)

    try:
        plan, plan_path, snapshots, paths, receipts = make_fixture(output / "synthetic_collect_campaign", script, manifest, coordinator)
        campaign = Path(plan["campaign_root"])
        before = sorted(str(path) for path in campaign.rglob("*"))
        validated = cli(plan_path, paths=paths, pin=manifest.digest(plan))
        require(validated["status"] == "validated_only" and validated["writes_performed"] is False, "Validate-only CLI failed")
        require(before == sorted(str(path) for path in campaign.rglob("*")), "Validate-only created a file or directory")
        cases.append({"case": "complete 57 validate-only CLI with zero writes", "status": "PASS"})
        cases.append(rejection("missing fifth snapshot", lambda: coordinator.collect(plan, paths[:-1], validate_only=True)))
        cases.append(rejection("duplicate producing host", lambda: coordinator.collect(plan, paths[:-1] + [paths[0]], validate_only=True)))
        cases.append(rejection("wrong independent plan pin", lambda: cli(plan_path, paths=paths, pin="0" * 64)))

        def variant(name, change):
            altered = copy.deepcopy(snapshots["a"])
            change(altered)
            path = campaign / "invalid_snapshot_inputs" / (name + ".json")
            manifest.frozen(path, altered)
            return [str(path), *paths[1:]]

        cases.append(rejection("snapshot different plan", lambda: coordinator.collect(plan,
            variant("wrong_plan", lambda value: value.update(plan_sha256="0" * 64)), validate_only=True)))
        cases.append(rejection("wrong producing hostname", lambda: coordinator.collect(plan,
            variant("wrong_host", lambda value: value.update(hostname="not-the-producing-host")), validate_only=True)))
        cases.append(rejection("incomplete lane", lambda: coordinator.collect(plan,
            variant("incomplete", lambda value: value.update(all_requested_complete=False)), validate_only=True)))
        cases.append(rejection("unverified row", lambda: coordinator.collect(plan,
            variant("unverified", lambda value: value["rows"][0].update(status="missing")), validate_only=True)))
        cases.append(rejection("duplicate row", lambda: coordinator.collect(plan,
            variant("duplicate_row", lambda value: value["rows"].__setitem__(1, value["rows"][0])), validate_only=True)))

        def wrong_artifacts(value):
            tid = value["rows"][0]["trial_id"]
            bad = copy.deepcopy(value["verified_receipts"][tid])
            bad["artifacts"] = list(reversed(bad["artifacts"]))
            value["verified_receipts"][tid] = bad
            value["rows"][0]["receipt"] = bad

        cases.append(rejection("wrong ordered eight detection artifacts", lambda: coordinator.collect(plan,
            variant("wrong_artifacts", wrong_artifacts), validate_only=True)))
        not_ready = copy.deepcopy(plan)
        not_ready["trials"][-1]["kind"] = "not_ready"
        cases.append(rejection("explicit not_ready seed0", lambda: coordinator.collect(not_ready, paths, validate_only=True), "not_ready"))
        reduced = copy.deepcopy(plan)
        removed = reduced["trials"].pop()["id"]
        reduced["seed0_trial_ids"].remove(removed)
        for lane in reduced["lanes"]:
            if removed in lane["trial_ids"]:
                lane["trial_ids"].remove(removed)
        cases.append(rejection("smaller 56-trial matrix", lambda: coordinator.collect(reduced, paths, validate_only=True), "57"))
        require(not Path(plan["barrier_path"]).exists() and not (campaign / "locks").exists()
                and not (campaign / "evidence").exists(), "Rejected/validate-only cases wrote publication state")
        publication = coordinator.collect(plan, paths, str(campaign / "evidence" / "collected_valid"))
        require(publication["status"] == "published", "Complete fixture did not publish")
        require(manifest.verify_barrier(plan)["receipts"] == receipts, "Published fixture lost or changed original receipt fields")
        again = coordinator.collect(plan, paths, str(campaign / "evidence" / "must_not_be_created"))
        require(again["status"] == "already_published" and not (campaign / "evidence" / "must_not_be_created").exists(),
                "Repeated same evidence did not preserve the existing barrier")
        cases.append({"case": "complete collection, unchanged issue_barrier, exact receipt preservation and idempotency", "status": "PASS"})

        plan2, plan2_path, unused_snapshots, unused_paths, receipts2 = make_fixture(
            output / "synthetic_accept_campaign", script, manifest, coordinator)
        payload = {"schema": manifest.SCHEMA, "state": "all_seed0_complete", "plan_sha256": manifest.digest(plan2),
                   "seed0_trial_ids": sorted(plan2["seed0_trial_ids"]), "receipts": receipts2,
                   "issued_utc": manifest.utc(), "trust_model": "synthetic fixture, trusted coordinator, not a signature"}
        marker = {"payload": payload, "payload_sha256": manifest.digest(payload)}
        candidate = Path(plan2["campaign_root"]) / "synthetic_candidate.json"
        manifest.frozen(candidate, marker)
        validated2 = cli(plan2_path, candidate=candidate, pin=manifest.digest(plan2))
        require(validated2["status"] == "validated_only" and not Path(plan2["barrier_path"]).exists(), "Acceptance validate-only wrote a barrier")
        accepted = coordinator.accept(plan2, str(candidate))
        require(accepted["status"] == "accepted" and manifest.verify_barrier(plan2)["receipts"] == receipts2, "Acceptance contract failed")
        barrier_sha = manifest.sha(plan2["barrier_path"])
        require(coordinator.accept(plan2, str(candidate))["status"] == "already_published", "Same candidate was not idempotent")
        cases.append({"case": "candidate CLI validate-only, accepted via frozen, existing verify_barrier, idempotency", "status": "PASS"})
        changed = copy.deepcopy(marker)
        changed["payload"]["issued_utc"] = "2026-10-09T01:02:03+00:00"
        changed["payload_sha256"] = manifest.digest(changed["payload"])
        different = Path(plan2["campaign_root"]) / "synthetic_different_valid_candidate.json"
        manifest.frozen(different, changed)
        cases.append(rejection("different valid candidate cannot overwrite", lambda: coordinator.accept(plan2, str(different)), "Frozen artifact differs"))
        partial = copy.deepcopy(marker)
        partial["payload"]["receipts"].pop(plan2["seed0_trial_ids"][-1])
        partial["payload_sha256"] = manifest.digest(partial["payload"])
        partial_path = Path(plan2["campaign_root"]) / "synthetic_partial_candidate.json"
        manifest.frozen(partial_path, partial)
        cases.append(rejection("partial candidate receipt coverage", lambda: coordinator.accept(plan2, str(partial_path))))
        require(manifest.sha(plan2["barrier_path"]) == barrier_sha, "Rejected candidate modified the existing barrier")
        require("torch" not in sys.modules, "Standard-library coordinator fixture imported torch")
        report.update(status="PASS", cases=cases, no_gpu_import=True, publication=publication,
                      accepted_fixture=accepted, actual_campaign_plan_or_barrier_touched=False)
    except BaseException as error:
        report.update(status="FAIL", cases=cases, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["finished_utc"] = manifest.utc()
        manifest.frozen(output / "coordinator_probe_report.json", report)
        print(json.dumps({"status": report["status"], "cases": len(cases), "report": str(output / "coordinator_probe_report.json")}))


if __name__ == "__main__":
    main()
