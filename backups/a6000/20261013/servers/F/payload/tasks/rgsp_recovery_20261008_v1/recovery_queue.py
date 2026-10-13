#!/usr/bin/env python3
"""Finite recovery queue after a specific pre-existing GPU queue completes.

Original experiment directories and the preceding workload are read-only.
Preparation/audit is CPU-only; training starts only after dependency + idle gates.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASKS = Path("/home/powerleader/ig-ds-ssm/tasks")
PYTHON = Path("/home/powerleader/ig-ds-ssm/envs/ig-ds-ssm-env/bin/python")
SSDD = ("original", "matched_identity", "matched_scale075", "local_energy",
        "local_energy_input", "local_energy_delta", "full", "full_fixed_half",
        "residual_scale075", "energy_only", "guidance_only", "global")
SADD = ("original", "matched_scale075", "local_energy", "full")
HOSTS = {
    "a": ("powerleader-PT6620W", "SADD", 0, 0, "GPU-12cf4841-46b9-94d1-2715-9932134cf964", "a6000_a"),
    "b": ("powerleader-a6000-b", "SADD", 1, 0, "GPU-09b018e1-7140-296a-73d4-9e010ee209a2", "a6000_b"),
    "d": ("a6000-d", "SADD", 2, 0, "GPU-f1b51ae3-c020-9901-c608-cca67f6ab8f1", "a6000_d"),
    "f0": ("powerleader-a6000-f", "SSDD", 0, 0, "GPU-9d27eec3-fd30-8534-8341-b983bea3832b", "a6000_f_gpu0"),
    "f1": ("powerleader-a6000-f", "SSDD", 1, 1, "GPU-ce57e76d-ee2e-ff91-dbdd-ec0938874c40", "a6000_f_gpu1"),
}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    p = Path(p)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, p)


def layout(key):
    host, dataset, seed, index, uuid, predecessor = HOSTS[key]
    if socket.gethostname() != host:
        raise RuntimeError("Unexpected host identity")
    campaign = "rgsp_controls_20261008_v3" if dataset == "SSDD" else "rgsp_sadd_controls_20261008_v1"
    old = TASKS / campaign / f"seed{seed}"
    output = ROOT / f"{dataset.lower()}_seed{seed}"
    pred = (Path("/home/powerleader/codex_sar_agent_final_v1p1_20261008/results/final/")
            / "study_v1_1/model_runs_dist" / predecessor / "frozen_queue_state.json")
    return {"key": key, "host": host, "dataset": dataset, "seed": seed,
            "gpu_index": index, "gpu_uuid": uuid, "source_lane": str(old),
            "code": str(old.parent / "code"), "output": str(output),
            "predecessor_state": str(pred), "variants": list(SSDD if dataset == "SSDD" else SADD),
            "epochs": 150 if dataset == "SSDD" else 200, "patience": 30 if dataset == "SSDD" else 40}


def code_audit(cfg):
    old, code = Path(cfg["source_lane"]), Path(cfg["code"])
    protocol = read(old / "manifest/protocol.json")
    assert protocol["variants"] == cfg["variants"] and protocol["seed"] == cfg["seed"]
    assert protocol["epochs"] == cfg["epochs"] and protocol["batch"] == 16
    for name, digest in protocol["code"].items():
        if sha(code / name) != digest:
            raise RuntimeError(f"Frozen code changed: {name}")
    for variant, digest in protocol["config_sha256"].items():
        assert sha(old / "manifest/configs" / (variant + ".yaml")) == digest
    assert sha(old / "manifest/data_content.json") == protocol["data_manifest_sha256"]
    return protocol


def completion(run, seed):
    train, evaluation = run / "train_complete.json", run / "eval_test.json"
    if not train.exists() or not evaluation.exists():
        return None
    t, e = read(train), read(evaluation)
    assert not t["smoke_only"] and e["seed"] == seed
    assert t["best_sha256"] == e["weights_sha256"] == sha(run / "weights/best.pt")
    return {"variant": run.name, "seed": seed, **e["overall"],
            "weights_sha256": e["weights_sha256"], "evidence": str(evaluation)}


def predecessor_ok(state, cfg):
    if state.get("queue_signature_sha256") != cfg["predecessor_signature"]:
        raise RuntimeError("Predecessor queue changed; refusing to bypass it")
    if state.get("gpu_uuid") != cfg["gpu_uuid"]:
        raise RuntimeError("Predecessor GPU changed")
    expected = {(r["seed"], r["arm"]) for r in state["queue"]}
    completed = {(r["seed"], r["arm"]) for r in state["completed"] if r["status"] == "complete"}
    return state["status"] in ("complete", "complete_shard") and completed == expected


def gpu_state(cfg):
    out = subprocess.check_output(["nvidia-smi", "-i", str(cfg["gpu_index"]),
        "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=15)
    uuid, memory, utilization = next(csv.reader([out.strip()], skipinitialspace=True))
    if uuid != cfg["gpu_uuid"]:
        raise RuntimeError("GPU UUID/index mapping changed")
    pids = subprocess.check_output(["nvidia-smi", "-i", uuid,
        "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=15).strip()
    return {"uuid": uuid, "memory_mib": int(memory), "utilization": int(utilization),
            "pids": pids.splitlines() if pids else []}


def gpu_idle(gpu):
    return not gpu["pids"] and gpu["memory_mib"] <= 2500 and gpu["utilization"] <= 10


def predecessor_process_alive(cfg):
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        tokens = [x.decode(errors="replace") for x in args]
        # Actual Python queue only, not an inherited shell command string.
        if any(x.endswith("/run_frozen_final_queue.py") for x in tokens) and cfg["gpu_uuid"] in tokens:
            return True
    return False


def cpu_checkpoint(path, cfg):
    import torch
    torch.set_num_threads(1)
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"Checkpoint CRC failure: {path.name}: {bad}")
    checkpoint = torch.load(path, map_location="cpu")
    epoch = checkpoint["epoch"] + 1
    args = checkpoint["train_args"]
    assert 0 < epoch < cfg["epochs"] and checkpoint["optimizer"] is not None
    assert args["seed"] == cfg["seed"] and args["epochs"] == cfg["epochs"]
    assert args["batch"] == 16 and args["optimizer"] == "AdamW"
    model = checkpoint.get("ema") or checkpoint.get("model")
    assert model is not None
    for name, tensor in model.state_dict().items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"Nonfinite checkpoint tensor: {name}")
    from train_rgsp_control import controls
    observed = controls(model)
    report = {"path": str(path), "sha256": sha(path), "completed_epoch": epoch,
              "optimizer_present": True, "controls": observed, "zip_crc": "PASS",
              "resume_semantics": "legacy EMA/FP16 optimizer; no bitwise/RNG replay guarantee"}
    del checkpoint, model
    return report


def clean_derived_records(run, checkpoint_epoch):
    changes = []
    for name in ("controls_by_epoch.jsonl", "results.csv"):
        p = run / name
        raw = p.read_bytes()
        cleaned = raw.rstrip(b"\0")
        if b"\0" in cleaned:
            raise RuntimeError(f"Embedded NUL in {p}; not safely repairable")
        lines = cleaned.decode().splitlines()
        if name.endswith("jsonl"):
            rows = [json.loads(line) for line in lines if line.strip()]
            kept = [r for r in rows if r["epoch"] <= checkpoint_epoch]
            payload = "".join(json.dumps(r) + "\n" for r in kept)
        else:
            kept = [line for line in lines[1:] if line.strip() and int(line.split(",")[0]) <= checkpoint_epoch]
            payload = "\n".join([lines[0]] + kept) + "\n"
        p.write_text(payload)
        changes.append({"file": name, "source_sha256": hashlib.sha256(raw).hexdigest(),
                        "derived_sha256": sha(p), "trailing_nul_bytes": len(raw) - len(cleaned),
                        "retained_through_epoch": checkpoint_epoch})
    return changes


def prepare(cfg):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Preparation must explicitly disable CUDA")
    old, output = Path(cfg["source_lane"]), Path(cfg["output"])
    if output.exists():
        raise RuntimeError("Recovery directory already exists; inspect, do not overwrite")
    assert shutil.disk_usage(ROOT).free > 20 * 1024 ** 3
    sys.path[:0] = [cfg["code"], str(Path(cfg["code"]) / "tools")]
    protocol = code_audit(cfg)
    pred = read(cfg["predecessor_state"])
    cfg["predecessor_signature"] = pred["queue_signature_sha256"]
    assert pred["gpu_uuid"] == cfg["gpu_uuid"]
    completed, pending = [], []
    checkpoint = None
    for variant in cfg["variants"]:
        run = old / "runs" / variant
        done = completion(run, cfg["seed"])
        if done:
            completed.append(done)
            continue
        pending.append(variant)
        if run.exists():
            if checkpoint is not None:
                raise RuntimeError("More than one partial run")
            checkpoint = cpu_checkpoint(run / "weights/last.pt", cfg)
            checkpoint["variant"] = variant
            # Validate the saved best as well; it remains the validation selection reference.
            checkpoint["best"] = cpu_checkpoint(run / "weights/best.pt", cfg)
    assert checkpoint is not None and pending[0] == checkpoint["variant"]
    output.mkdir()
    shutil.copytree(old / "manifest", output / "manifest")
    shutil.copytree(old / "data", output / "data")
    import yaml
    data = yaml.safe_load((output / "manifest/data.yaml").read_text())
    data["path"] = str(output / "data")
    (output / "manifest/data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    for row in read(output / "manifest/data_content.json"):
        # Both dataset manifests use split/name plus image/label SHA256.
        image_name = row.get("name", row.get("image"))
        if not isinstance(image_name, str):
            raise RuntimeError("Unknown data manifest row shape")
        assert sha(output / "data/images" / row["split"] / image_name) == row["image_sha256"]
        assert sha(output / "data/labels" / row["split"] / (Path(image_name).stem + ".txt")) == row["label_sha256"]
    variant = checkpoint["variant"]
    derived = output / "runs" / variant
    shutil.copytree(old / "runs" / variant, derived)
    cleaned = clean_derived_records(derived, checkpoint["completed_epoch"])
    cfg.update({"pending": pending, "source_completed": completed,
                "checkpoint": checkpoint, "record_cleanup": cleaned,
                "protocol_sha256": sha(old / "manifest/protocol.json"),
                "prepared_unix": time.time(), "original_artifacts": {}})
    for run in (old / "runs").iterdir():
        for file in run.rglob("*"):
            if file.is_file():
                cfg["original_artifacts"][str(file.relative_to(old))] = sha(file)
    cfg["runner_sha256"] = sha(Path(__file__))
    cfg["redirect_sha256"] = sha(ROOT / "train_redirect.py")
    write(output / "recovery_manifest.json", cfg)
    write(output / "queue_state.json", {"state": "prepared", "pending": pending,
          "completed": [r["variant"] for r in completed], "checkpoint_epoch": checkpoint["completed_epoch"]})
    print(json.dumps({"key": cfg["key"], "status": "prepared", "checkpoint_epoch": checkpoint["completed_epoch"],
                      "pending": pending, "output": str(output), "zip_crc": "PASS"}))


def queue(cfg):
    output = Path(cfg["output"])
    cfg = read(output / "recovery_manifest.json")
    assert sha(Path(__file__)) == cfg["runner_sha256"]
    assert sha(ROOT / "train_redirect.py") == cfg["redirect_sha256"]
    locks = []
    for path in (output / "queue.lock", Path(cfg["source_lane"]) / "lane.lock"):
        lock = path.open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    code_audit(cfg)
    state = {"state": "waiting_predecessor", "pid": os.getpid(), "host": socket.gethostname(),
             "gpu_uuid": cfg["gpu_uuid"], "pending": cfg["pending"],
             "completed": [r["variant"] for r in cfg["source_completed"]],
             "predecessor_state": cfg["predecessor_state"], "started_unix": time.time()}

    def update(**values):
        state.update(values, updated_unix=time.time())
        write(output / "queue_state.json", state)

    def wait_ready():
        idle_since = None
        while True:
            pred = read(cfg["predecessor_state"])
            ready = predecessor_ok(pred, cfg) and not predecessor_process_alive(cfg)
            if not ready:
                idle_since = None
                update(state="waiting_predecessor", predecessor_status=pred["status"],
                       predecessor_completed=len(pred["completed"]), predecessor_total=len(pred["queue"]))
            else:
                gpu = gpu_state(cfg)
                idle_since = (idle_since or time.monotonic()) if gpu_idle(gpu) else None
                update(state="waiting_gpu_idle", gpu=gpu,
                       idle_seconds=0 if idle_since is None else time.monotonic() - idle_since)
                if idle_since is not None and time.monotonic() - idle_since >= 120:
                    return
            time.sleep(30)

    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": cfg["gpu_uuid"], "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "PYTHONPATH": cfg["code"], "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                "YOLO_AUTOINSTALL": "false", "WANDB_MODE": "disabled",
                "YOLO_CONFIG_DIR": str(output / "runtime-config"), "MPLCONFIGDIR": str(output / "matplotlib")})
    rows = list(cfg["source_completed"])

    def execute(command, log):
        code_audit(cfg)
        if not gpu_idle(gpu_state(cfg)):
            raise RuntimeError("GPU changed after gate; refusing to start")
        update(state="running", command=command)
        with log.open("a") as f:
            f.write("\n" + json.dumps({"time": time.time(), "command": command}) + "\n")
            f.flush()
            subprocess.run(command, cwd=cfg["code"], env=env, stdout=f, stderr=subprocess.STDOUT, check=True)

    try:
        (output / "logs").mkdir(exist_ok=True)
        update()
        for variant in cfg["pending"]:
            update(current=variant)
            wait_ready()
            run = output / "runs" / variant
            cmd = [str(PYTHON), str(ROOT / "train_redirect.py"), "--code", cfg["code"],
                   "--output", str(output), "--variant", variant, "--seed", str(cfg["seed"]),
                   "--epochs", str(cfg["epochs"]), "--patience", str(cfg["patience"])]
            if variant == cfg["checkpoint"]["variant"]:
                assert sha(run / "weights/last.pt") == cfg["checkpoint"]["sha256"]
                cmd.append("--resume")
            execute(cmd, output / "logs" / (variant + ".train.log"))
            train = read(run / "train_complete.json")
            assert sha(run / "weights/best.pt") == train["best_sha256"]
            wait_ready()
            cmd = [str(PYTHON), str(Path(cfg["code"]) / "tools/evaluate_sarmix.py"),
                   "--weights", str(run / "weights/best.pt"), "--data", str(output / "manifest/data.yaml"),
                   "--experiment", f"rgsp_recovery_{cfg['dataset']}_seed{cfg['seed']}_{variant}",
                   "--placement", "baseline" if variant == "original" else "p3", "--variant", variant,
                   "--state-dim", "0" if variant == "original" else "16", "--seed", str(cfg["seed"]),
                   "--subset", cfg["dataset"], "--split", "test", "--output-json", str(run / "eval_test.json"),
                   "--imgsz", "640", "--batch", "16", "--device", "0", "--workers", "0", "--conf", "0.001",
                   "--iou", "0.6", "--project", str(output / "evaluations"), "--name", variant]
            execute(cmd, output / "logs" / (variant + ".eval.log"))
            row = completion(run, cfg["seed"])
            assert row is not None
            write(run / "evaluation_complete.json", {"exit_code": 0, "eval_sha256": sha(run / "eval_test.json"),
                  "finished_unix": time.time(), "smoke_only": False})
            rows.append(row)
            write(output / "results_partial.json", rows)
            update(completed=[r["variant"] for r in rows])
        assert len(rows) == len(cfg["variants"])
        for name, digest in cfg["original_artifacts"].items():
            assert sha(Path(cfg["source_lane"]) / name) == digest
        update(state="complete", finished_unix=time.time(), original_artifacts_unchanged=True)
    except BaseException as error:
        update(state="failed", error=repr(error))
        raise


def main():
    sys.dont_write_bytecode = True
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("prepare", "queue"))
    p.add_argument("--key", choices=HOSTS, required=True)
    args = p.parse_args()
    cfg = layout(args.key)
    (prepare if args.mode == "prepare" else queue)(cfg)


if __name__ == "__main__":
    main()
