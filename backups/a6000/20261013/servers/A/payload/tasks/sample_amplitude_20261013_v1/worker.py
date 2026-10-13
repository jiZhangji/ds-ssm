#!/usr/bin/env python3
"""Independent SAMPLE amplitude single-GPU lane worker (standard library only).

CLI: worker.py --plan lane.json --root /absolute/fresh/task --lane a0
     worker.py --plan lane.json --validate-plan

Plan v1: {version: 1, lane: str, gpu_uuid: "GPU-...", jobs: [...]}
Each job has id, task, dataset, method, command_probe and command_train.
Commands are argv arrays, never shell strings. Their only template fields are
{seed}, {output}, {gpu_uuid}. Train must use {seed}; both must use {output}.
Optional job fields: stage (0 CD, 10 classification, 20 SSDD, 30 SADD, 90 Ship),
estimated_seconds, min_free_gb (GPU free GiB), protocol_sha256, cwd, env,
blocked_reason. min_disk_free_gb is a lane field, default/minimum 15 GiB.

Outputs: root/lanes/LANE/jobs/JOB/{probe,run0,run1,run2}/. Probe writes result.json
or a final single-line JSON on stdout: status=PROBE_PASS, test_evaluated=false,
estimated_full_seconds>0; optional protocol_sha256 is seed-independent.
Training writes result.json: status=TRAINING_COMPLETE, integer seed, nonempty
finite numeric metrics, seed-independent protocol_sha256, checkpoint={path,
sha256}. Checkpoint must be a file inside that run's output. A completion.json
must contain status=TRAINING_COMPLETE and result_sha256 of result.json bytes.
The flat checkpoint_path/checkpoint_sha256 spelling is accepted. An optional
contract adds required_fields (dotted paths) and changes result_filename or
completion_filename. It cannot remove the baseline checks. --print-contract
prints the machine-readable baseline. Probe never substitutes for training.

Each lane and GPU has a new-task flock. The existing old-campaign GPU flock
is also mandatory; its contents are never written, nor is it created.
Locks are inherited by children and only CLOSED, never explicitly unlocked.
Before every process, require >=30 continuous seconds with no GPU compute
processes, low utilization, enough GPU memory and disk. Busy/unknown resources
wait; there is no CPU fallback, timeout, preemption, retry or remote command.

Scheduling is lane-local: stage, then probe/all seed0/all seed1/all seed2,
then the literal job order in the plan. Every job has exactly seeds 0, 1, 2;
measured or estimated duration never drops a seed or changes plan priority.
Accuracy is never read by scheduling. Failed or interrupted attempts block the
job, not the whole lane. Re-entry verifies completed outputs; uncertain work is
not retried. SIGINT/SIGTERM request a stop after the active child, without
signalling that child. State and JSONL events are individually atomic snapshots.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import string
import stat
import subprocess
import sys
import tempfile
import time

try:
    import fcntl
except ImportError:  # Static plan validation remains portable; execution is Linux.
    fcntl = None


LEGACY_ROOT = Path('/home/powerleader/ig-ds-ssm/tasks/rgsp_seedwise_20261009_v1')
TASK_ROOT = Path(__file__).resolve().parent
FIXED_SEEDS = (0, 1, 2)
SAFE_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$')
SHA256 = re.compile(r'^[0-9a-fA-F]{64}$')
BASE_CONTRACT = {
    'probe': {'status': 'PROBE_PASS', 'test_evaluated': False,
              'estimated_full_seconds': 'finite number > 0'},
    'train': {'status': 'TRAINING_COMPLETE', 'seed': 'integer matching requested seed',
              'metrics': 'nonempty finite numeric mapping',
              'protocol_sha256': '64 hex; identical across this job\'s seeds',
              'checkpoint': {'path': 'file inside output', 'sha256': '64 hex'}},
    'completion': {'status': 'TRAINING_COMPLETE', 'result_sha256': 'SHA256 of result.json bytes'},
}


class WorkerError(RuntimeError):
    pass


class ResourceUnavailable(WorkerError):
    pass


def positive(value, label, minimum=0.0, inclusive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise WorkerError(label + ' must be a finite number')
    if value < minimum or (not inclusive and value == minimum):
        raise WorkerError(label + ' is below the allowed minimum')
    return float(value)


def safe_name(value, label):
    if not isinstance(value, str) or not SAFE_NAME.fullmatch(value):
        raise WorkerError(label + ' must be a simple safe name')
    return value


def valid_sha(value, label):
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise WorkerError(label + ' must be a SHA256 hex digest')
    return value.lower()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open('r', encoding='utf-8-sig') as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise WorkerError(str(path) + ' must contain a JSON object')
    return value


def inside(path, root):
    candidate, base = Path(path).resolve(), Path(root).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        raise WorkerError('path escapes the task directory: ' + str(path))
    if candidate == base:
        raise WorkerError('artifact cannot be the task directory itself')
    return candidate


def atomic_bytes(path, data, overwrite=True):
    """Durable rename, or hard-link publication for no-clobber initialization."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            os.unlink(temporary)
        if os.name == 'posix':
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)  # Only this function's exact fresh temporary file.


def atomic_json(path, value, overwrite=True):
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
                 .encode('utf-8'), overwrite=overwrite)


def validate_command(command, label, training=False):
    if not isinstance(command, list) or not command or any(
            not isinstance(arg, str) or not arg or '\x00' in arg or '\n' in arg for arg in command):
        raise WorkerError(label + ' must be a nonempty argv array')
    fields = set()
    for arg in command:
        try:
            pieces = list(string.Formatter().parse(arg))
        except ValueError as exc:
            raise WorkerError(label + ': invalid template') from exc
        for _, field, fmt, conversion in pieces:
            if field is not None:
                if field not in {'seed', 'output', 'gpu_uuid'} or fmt or conversion:
                    raise WorkerError(label + ': unsupported template field')
                fields.add(field)
    if 'output' not in fields or (training and 'seed' not in fields):
        raise WorkerError(label + ' must bind {output}' + (' and {seed}' if training else ''))


def default_stage(job):
    task = str(job.get('task', '')).lower().replace('-', '_')
    dataset = str(job.get('dataset', '')).lower()
    if task in {'cd', 'change_detection', 'change detection'}:
        return 0
    if task in {'class', 'classification', 'cls'}:
        return 10
    if 'ssdd' in dataset:
        return 20
    if 'sadd' in dataset:
        return 30
    return 90


def validate_plan(raw, lane=None):
    if not isinstance(raw, dict) or type(raw.get('version')) is not int or raw['version'] != 1:
        raise WorkerError('plan version must be 1')
    plan = copy.deepcopy(raw)
    safe_name(plan.get('lane'), 'lane')
    if lane is not None and plan['lane'] != lane:
        raise WorkerError('--lane does not match plan lane')
    gpu = plan.get('gpu_uuid')
    if not isinstance(gpu, str) or not re.fullmatch(r'GPU-[A-Za-z0-9-]+', gpu):
        raise WorkerError('gpu_uuid must be a full GPU-... identifier')
    required = plan.setdefault('legacy_lock_required', True)
    if required is not True:
        raise WorkerError('this task requires the existing legacy GPU lock')
    canonical_lock = str(LEGACY_ROOT / 'locks' / (gpu + '.lock'))
    plan.setdefault('legacy_lock', canonical_lock if required else None)
    if required and plan['legacy_lock'] != canonical_lock:
        raise WorkerError('required legacy_lock must be the registered old-campaign GPU lock')
    seeds = plan.setdefault('seeds', list(FIXED_SEEDS))
    if not isinstance(seeds, list) or any(type(seed) is not int for seed in seeds) or seeds != list(FIXED_SEEDS):
        raise WorkerError('this task requires exactly seeds [0, 1, 2]')
    if plan.setdefault('seed_policy', 'fixed_three') != 'fixed_three':
        raise WorkerError('this task requires seed_policy=fixed_three')
    plan['min_disk_free_gb'] = positive(plan.get('min_disk_free_gb', 15), 'min_disk_free_gb', 15, True)
    plan['idle_seconds'] = positive(plan.get('idle_seconds', 30), 'idle_seconds', 30, True)
    plan['poll_seconds'] = positive(plan.get('poll_seconds', 5), 'poll_seconds', 0.2, True)
    if plan['poll_seconds'] > 30:
        raise WorkerError('poll_seconds may not exceed 30')
    plan['max_gpu_utilization'] = positive(plan.get('max_gpu_utilization', 5), 'max_gpu_utilization', 0, True)
    if plan['max_gpu_utilization'] > 10:
        raise WorkerError('max_gpu_utilization may not exceed 10 percent')
    contract = plan.setdefault('contract', {})
    if not isinstance(contract, dict):
        raise WorkerError('contract must be an object')
    for field, default in [('result_filename', 'result.json'), ('completion_filename', 'completion.json')]:
        safe_name(contract.setdefault(field, default), field)
    if contract['result_filename'] == contract['completion_filename']:
        raise WorkerError('result and completion must be distinct files')
    fields = contract.setdefault('required_fields', [])
    if not isinstance(fields, list) or any(not isinstance(f, str) or not re.fullmatch(
            r'[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*', f) for f in fields):
        raise WorkerError('required_fields must be dotted field names')
    jobs = plan.get('jobs')
    if not isinstance(jobs, list) or not jobs or len(jobs) > 2048:
        raise WorkerError('jobs must contain 1 to 2048 finite jobs')
    seen = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise WorkerError('every job must be an object')
        identifier = safe_name(job.get('id'), 'job id')
        if identifier in seen:
            raise WorkerError('duplicate job id: ' + identifier)
        seen.add(identifier)
        for field in ('task', 'dataset', 'method'):
            if not isinstance(job.get(field), str) or not job[field].strip():
                raise WorkerError(identifier + ': missing ' + field)
        job.setdefault('stage', default_stage(job))
        if type(job['stage']) is not int or job['stage'] < 0:
            raise WorkerError(identifier + ': stage must be a nonnegative integer')
        job['estimated_seconds'] = positive(job.get('estimated_seconds', 3600), 'estimated_seconds')
        job['min_free_gb'] = positive(job.get('min_free_gb', 8), 'min_free_gb', 0, True)
        if 'protocol_sha256' in job:
            job['protocol_sha256'] = valid_sha(job['protocol_sha256'], 'protocol_sha256')
        if 'cwd' in job and (not isinstance(job['cwd'], str) or not job['cwd'].startswith('/')):
            raise WorkerError(identifier + ': cwd must be an absolute Linux path')
        env = job.get('env', {})
        if not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                or '\x00' in k + v or '=' in k or k in {'HOME', 'CODEX_HOME', 'CUDA_VISIBLE_DEVICES'}
                for k, v in env.items()):
            raise WorkerError(identifier + ': invalid environment overrides')
        if 'blocked_reason' in job:
            if not isinstance(job['blocked_reason'], str) or not job['blocked_reason'].strip():
                raise WorkerError(identifier + ': blocked_reason must be explicit text')
        else:
            validate_command(job.get('command_probe'), identifier + '.command_probe')
            validate_command(job.get('command_train'), identifier + '.command_train', training=True)
    return plan


def new_state(plan, root):
    jobs = {}
    for job in plan['jobs']:
        jobs[job['id']] = {
            'status': 'BLOCKED' if job.get('blocked_reason') else 'PENDING',
            'reason': job.get('blocked_reason'), 'estimated_seconds': job['estimated_seconds'],
            'seed_limit': len(FIXED_SEEDS), 'protocol_sha256': job.get('protocol_sha256'),
            'probe': {'status': 'PENDING'},
            'runs': {str(seed): {'status': 'PENDING'} for seed in range(3)},
        }
    return {'version': 1, 'lane': plan['lane'], 'gpu_uuid': plan['gpu_uuid'],
            'root': str(root), 'plan_sha256': canonical_sha(plan), 'revision': 0,
            'status': 'READY', 'jobs': jobs}


def next_action(plan, state, probe_only=False):
    """Pure scheduler. Metrics/accuracy are deliberately never consulted."""
    candidates = []
    for position, job in enumerate(plan['jobs']):
        current = state['jobs'][job['id']]
        if current['status'] in {'BLOCKED', 'COMPLETE'}:
            continue
        if current['probe']['status'] == 'PENDING':
            kind, seed, round_number = 'probe', 0, -1
        elif current['probe']['status'] == 'COMPLETE':
            if probe_only:
                continue
            seeds = [s for s in FIXED_SEEDS if current['runs'][str(s)]['status'] == 'PENDING']
            if not seeds:
                continue
            seed = seeds[0]
            if any(current['runs'][str(s)]['status'] != 'COMPLETE' for s in range(seed)):
                continue
            kind, round_number = 'train', seed
        else:
            continue
        key = (job['stage'], round_number, position)
        candidates.append((key, job, kind, seed))
    if not candidates:
        return None
    _, job, kind, seed = min(candidates, key=lambda item: item[0])
    return job, kind, seed


def numeric_metrics(value):
    if isinstance(value, dict):
        return bool(value) and all(numeric_metrics(v) for v in value.values())
    if isinstance(value, list):
        return bool(value) and all(numeric_metrics(v) for v in value)
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def validate_probe(payload):
    if not isinstance(payload, dict) or payload.get('status') != 'PROBE_PASS':
        raise WorkerError('probe did not report PROBE_PASS')
    if payload.get('test_evaluated') is not False:
        raise WorkerError('probe must explicitly report test_evaluated=false')
    positive(payload.get('estimated_full_seconds'), 'estimated_full_seconds')
    if payload.get('device') == 'cpu':
        raise WorkerError('CPU probe is prohibited')
    if 'protocol_sha256' in payload:
        valid_sha(payload['protocol_sha256'], 'probe protocol_sha256')
    return payload


def validate_training(output, seed, contract, expected_protocol=None, job=None):
    output = Path(output).resolve()
    result_path = inside(output / contract.get('result_filename', 'result.json'), output)
    completion_path = inside(output / contract.get('completion_filename', 'completion.json'), output)
    result, completion = read_json(result_path), read_json(completion_path)
    if result.get('status') != 'TRAINING_COMPLETE' or completion.get('status') != 'TRAINING_COMPLETE':
        raise WorkerError('both result and completion must report TRAINING_COMPLETE')
    if type(result.get('seed')) is not int or result['seed'] != seed:
        raise WorkerError('result seed does not match the requested seed')
    if not isinstance(result.get('metrics'), dict) or not numeric_metrics(result['metrics']):
        raise WorkerError('metrics must be a nonempty finite numeric mapping')
    protocol = valid_sha(result.get('protocol_sha256'), 'result protocol_sha256')
    if expected_protocol is not None and protocol != valid_sha(expected_protocol, 'expected protocol'):
        raise WorkerError('protocol digest differs from this job\'s frozen protocol')
    if result.get('device') == 'cpu':
        raise WorkerError('CPU training is prohibited')
    if job:
        for field in ('task', 'dataset', 'method'):
            if field in result and result[field] != job[field]:
                raise WorkerError('result identity mismatch: ' + field)
    for field in contract.get('required_fields', []):
        value = result
        for component in field.split('.'):
            if not isinstance(value, dict) or component not in value:
                raise WorkerError('missing required result field: ' + field)
            value = value[component]
        if value is None:
            raise WorkerError('null required result field: ' + field)
    checkpoint = result.get('checkpoint', {
        'path': result.get('checkpoint_path'), 'sha256': result.get('checkpoint_sha256')})
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('path'), str):
        raise WorkerError('checkpoint path/sha256 are required')
    path = Path(checkpoint['path'])
    checkpoint_path = inside(path if path.is_absolute() else output / path, output)
    if not checkpoint_path.is_file():
        raise WorkerError('checkpoint file is absent')
    checkpoint_sha = valid_sha(checkpoint.get('sha256'), 'checkpoint sha256')
    if file_sha(checkpoint_path) != checkpoint_sha:
        raise WorkerError('checkpoint SHA256 mismatch')
    result_sha = file_sha(result_path)
    if valid_sha(completion.get('result_sha256'), 'completion result_sha256') != result_sha:
        raise WorkerError('completion does not bind the result bytes')
    return {'result_sha256': result_sha, 'completion_sha256': file_sha(completion_path),
            'checkpoint_sha256': checkpoint_sha, 'checkpoint_path': str(checkpoint_path),
            'protocol_sha256': protocol, 'metrics': result['metrics']}


class HeldLock:
    def __init__(self, path, create):
        self.path, self.create, self.fd = Path(path), create, None

    def try_acquire(self):
        if fcntl is None:
            raise WorkerError('flock execution requires Linux')
        flags = os.O_RDWR | (os.O_CREAT if self.create else 0) | getattr(os, 'O_NOFOLLOW', 0)
        fd = os.open(str(self.path), flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise WorkerError('lock must be an existing regular file: ' + str(self.path))
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        except BaseException:
            os.close(fd)
            raise
        os.set_inheritable(fd, True)
        self.fd = fd
        return True

    def close(self):
        if self.fd is not None:
            # NEVER LOCK_UN: a surviving child may share this open file description.
            os.close(self.fd)
            self.fd = None


def query_gpu(gpu_uuid):
    def query(arguments):
        try:
            process = subprocess.run(['nvidia-smi'] + arguments, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, check=True, timeout=10)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ResourceUnavailable('GPU query failed: ' + type(exc).__name__) from exc
        return process.stdout.strip()
    text = query(['--id=' + gpu_uuid,
                  '--query-gpu=uuid,name,memory.free,memory.total,utilization.gpu',
                  '--format=csv,noheader,nounits'])
    rows = list(csv.reader(text.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 5 or rows[0][0].strip() != gpu_uuid:
        raise ResourceUnavailable('GPU identity query is ambiguous')
    uuid, name, free, total, utilization = [cell.strip() for cell in rows[0]]
    if 'A6000' not in name:
        raise ResourceUnavailable('assigned GPU is not an A6000')
    try:
        free, total, utilization = float(free), float(total), float(utilization)
        if not all(math.isfinite(x) for x in (free, total, utilization)) or not (
                0 <= free <= total and total > 0 and 0 <= utilization <= 100):
            raise ValueError('invalid GPU counters')
    except ValueError as exc:
        raise ResourceUnavailable('GPU counters unavailable') from exc
    compute = query(['--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'])
    pids = []
    for row in csv.reader(compute.splitlines(), skipinitialspace=True):
        if not row or (len(row) == 1 and not row[0].strip()):
            continue
        if len(row) != 2:
            raise ResourceUnavailable('compute process query is unavailable')
        if row[0].strip() == uuid:
            try:
                pid = int(row[1])
                if pid <= 0:
                    raise ValueError('invalid PID')
                pids.append(pid)
            except ValueError as exc:
                raise ResourceUnavailable('invalid compute PID') from exc
    return {'gpu_uuid': uuid, 'name': name, 'free_gb': free / 1024,
            'total_gb': total / 1024, 'utilization': utilization, 'compute_pids': pids}


class IdleWindow:
    def __init__(self, seconds=30):
        self.seconds, self.since = max(30, seconds), None

    def observe(self, now, ready):
        if not ready:
            self.since = None
            return False
        if self.since is None:
            self.since = now
        return now - self.since >= self.seconds


class Worker:
    def __init__(self, plan, root):
        raw_root = Path(root)
        if not raw_root.is_absolute():
            raise WorkerError('--root must be absolute')
        self.plan, self.root = plan, raw_root.resolve()
        if self.root != TASK_ROOT and TASK_ROOT not in self.root.parents:
            raise WorkerError('--root must remain inside this independent task directory')
        if self.root in {Path('/'), Path.home().resolve()} or self.root == LEGACY_ROOT or LEGACY_ROOT in self.root.parents:
            raise WorkerError('--root must be a fresh task, never home/root/old campaign')
        self.lane_root = inside(self.root / 'lanes' / plan['lane'], self.root)
        self.state_path = inside(self.lane_root / 'state.json', self.root)
        self.events_path = inside(self.lane_root / 'event.jsonl', self.root)
        self.state = None
        self.locks = []
        self.stop_requested = False
        self.probe_only = False
        self.gpu_inspector = query_gpu
        self.clock = time.monotonic
        self.sleep = time.sleep

    def stop(self, *_):
        self.stop_requested = True

    def persist(self, event, **detail):
        self.state['revision'] += 1
        self.state['updated_at'] = time.time()
        atomic_json(inside(self.state_path, self.root), self.state)
        record = {'revision': self.state['revision'], 'at': self.state['updated_at'],
                  'event': event, 'lane': self.plan['lane'], **detail}
        previous = self.events_path.read_bytes() if self.events_path.exists() else b''
        atomic_bytes(inside(self.events_path, self.root), previous + (
            json.dumps(record, sort_keys=True, allow_nan=False) + '\n').encode('utf-8'))

    def acquire(self, path, create, report=True):
        lock = HeldLock(path, create)
        waiting = False
        while not self.stop_requested:
            if lock.try_acquire():
                self.locks.append(lock)
                return True
            if not waiting:
                if report and self.state is not None:
                    self.state['status'] = 'WAITING_LOCK'
                    self.persist('waiting_lock', lock=str(path))
                else:
                    print('WAITING_LOCK ' + str(path), flush=True)
                waiting = True
            self.sleep(self.plan['poll_seconds'])
        return False

    def initialize_state(self):
        if self.state_path.exists():
            state = read_json(self.state_path)
            expected = new_state(self.plan, self.root)
            if any(state.get(k) != expected[k] for k in ('version', 'plan_sha256', 'lane', 'gpu_uuid', 'root')):
                raise WorkerError('existing state belongs to a different plan/root; refusing overwrite')
            if set(state.get('jobs', {})) != set(expected['jobs']):
                raise WorkerError('existing state job set is invalid')
            for identifier, current in state['jobs'].items():
                if current.get('seed_limit') != len(FIXED_SEEDS) or set(current.get('runs', {})) != {'0', '1', '2'}:
                    raise WorkerError(identifier + ': existing state violates the fixed-three-seed policy')
                if current.get('status') != 'BLOCKED' and any(
                        run.get('status') == 'SKIPPED' for run in current['runs'].values()):
                    raise WorkerError(identifier + ': an unblocked job may not skip a requested seed')
            self.state = state
        else:
            leftovers = [p.name for p in self.lane_root.iterdir() if p.name != 'worker.lock']
            if leftovers:
                raise WorkerError('lane without state contains prior files; refusing adoption/overwrite')
            atomic_json(self.lane_root / 'plan.frozen.json', self.plan, overwrite=False)
            self.state = new_state(self.plan, self.root)
            self.persist('initialized')

    def output_dir(self, job, kind, seed):
        return inside(self.lane_root / 'jobs' / job['id'] / ('probe' if kind == 'probe' else 'run' + str(seed)), self.root)

    def block(self, job, kind, seed, reason):
        current = self.state['jobs'][job['id']]
        record = current['probe'] if kind == 'probe' else current['runs'][str(seed)]
        record.update(status='BLOCKED', reason=reason)
        current.update(status='BLOCKED', reason=reason)
        for entry in current['runs'].values():
            if entry['status'] == 'PENDING':
                entry.update(status='SKIPPED', reason='job_blocked_no_automatic_retry')
        self.persist('job_blocked', job=job['id'], kind=kind, seed=seed, reason=reason)

    def log_path(self, job, kind, seed):
        output = self.output_dir(job, kind, seed)
        return inside(output.parent / (output.name + '.run.log'), self.root)

    def probe_payload(self, output, log_path):
        path = output / self.plan['contract']['result_filename']
        if path.exists():
            return validate_probe(read_json(inside(path, self.root)))
        with log_path.open('rb') as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 1024 * 1024))
            lines = handle.read().decode('utf-8', errors='replace').splitlines()
        lines = [line for line in lines if line.strip()]
        if not lines:
            raise WorkerError('probe did not produce JSON')
        return validate_probe(json.loads(lines[-1]))

    def finish(self, job, kind, seed, record, recovering=False):
        current, output = self.state['jobs'][job['id']], self.output_dir(job, kind, seed)
        if record.get('returncode', 0) != 0:
            raise WorkerError('process exit code ' + str(record['returncode']))
        if kind == 'probe':
            payload = self.probe_payload(output, self.log_path(job, kind, seed))
            if record.get('payload') is not None and record['payload'] != payload:
                raise WorkerError('previously verified probe payload has changed')
            current['estimated_seconds'] = float(payload['estimated_full_seconds'])
            protocol = payload.get('protocol_sha256')
            if protocol:
                protocol = valid_sha(protocol, 'probe protocol')
                if current['protocol_sha256'] and protocol != current['protocol_sha256']:
                    raise WorkerError('probe protocol mismatch')
                current['protocol_sha256'] = protocol
            record['payload'] = payload
        else:
            evidence = validate_training(output, seed, self.plan['contract'], current['protocol_sha256'], job)
            previous = record.get('verification')
            if previous and evidence != previous:
                raise WorkerError('previously verified artifacts have changed')
            record['verification'] = evidence
            current['protocol_sha256'] = evidence['protocol_sha256']
        record.update(status='COMPLETE', recovered=bool(recovering))
        current['status'] = 'ACTIVE'
        if kind == 'train' and all(r['status'] == 'COMPLETE' for r in current['runs'].values()):
            current['status'] = 'COMPLETE'
        self.persist('recovered_complete' if recovering else 'action_complete', job=job['id'], kind=kind, seed=seed)

    def reconcile(self):
        for job in self.plan['jobs']:
            current = self.state['jobs'][job['id']]
            if current['status'] == 'BLOCKED':
                continue
            records = [('probe', 0, current['probe'])] + [('train', s, current['runs'][str(s)]) for s in range(3)]
            for kind, seed, record in records:
                if record['status'] not in {'RUNNING', 'COMPLETE'}:
                    continue
                try:
                    self.finish(job, kind, seed, record, recovering=True)
                except (OSError, ValueError, WorkerError) as exc:
                    self.block(job, kind, seed, 'recovery_not_retried: ' + str(exc))
                    break

    def wait_ready(self, job):
        window, last_reason = IdleWindow(self.plan['idle_seconds']), None
        while not self.stop_requested:
            try:
                snapshot = self.gpu_inspector(self.plan['gpu_uuid'])
                disk_gb = shutil.disk_usage(self.root).free / (1024 ** 3)
                ready = (not snapshot['compute_pids'] and snapshot['utilization'] <= self.plan['max_gpu_utilization']
                         and snapshot['free_gb'] >= job['min_free_gb'] and disk_gb >= self.plan['min_disk_free_gb'])
                reason = 'continuous_idle_window' if ready else 'gpu_or_disk_busy'
                detail = {'gpu': snapshot, 'disk_free_gb': disk_gb}
            except (OSError, ResourceUnavailable) as exc:
                ready, reason, detail = False, 'resource_query_unavailable', {'error': str(exc)}
            if reason != last_reason:
                self.state['status'] = 'WAITING_RESOURCE'
                self.persist('waiting_resource', job=job['id'], reason=reason, **detail)
                last_reason = reason
            if window.observe(self.clock(), ready):
                self.persist('resource_ready', job=job['id'], **detail)
                return True
            self.sleep(self.plan['poll_seconds'])
        return False

    def execute(self, job, kind, seed):
        current = self.state['jobs'][job['id']]
        record = current['probe'] if kind == 'probe' else current['runs'][str(seed)]
        output = self.output_dir(job, kind, seed)
        process = None
        try:
            if output.exists():
                raise WorkerError('preexisting output; no overwrite or automatic adoption')
            output.parent.mkdir(parents=True, exist_ok=True)
            # The runner creates output atomically; keep captured stdout outside it.
            log_path = self.log_path(job, kind, seed)
            command = [arg.format(seed=seed, output=str(output), gpu_uuid=self.plan['gpu_uuid'])
                       for arg in job['command_' + kind]]
            record.update(status='RUNNING', started_at=time.time(), command=command,
                          output=str(output), log_path=str(log_path))
            self.state['status'], current['status'] = 'RUNNING', 'ACTIVE'
            self.persist('launch_intent', job=job['id'], kind=kind, seed=seed, command=command)
            env = os.environ.copy()
            env.update(job.get('env', {}))
            env.update(CUDA_VISIBLE_DEVICES=self.plan['gpu_uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       RGSP_GPU_UUID=self.plan['gpu_uuid'], RGSP_REQUIRE_CUDA='1',
                       PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       WANDB_MODE='disabled', WANDB_DISABLED='true',
                       XDG_CACHE_HOME=str(self.lane_root / 'cache'),
                       MPLCONFIGDIR=str(self.lane_root / 'cache' / 'matplotlib'))
            fds = tuple(lock.fd for lock in self.locks if lock.fd is not None)
            if len(fds) != 3:
                raise WorkerError('lane, task GPU, and existing legacy GPU locks must be held before launch')
            started = self.clock()
            with log_path.open('xb') as log:
                process = subprocess.Popen(command, cwd=job.get('cwd', str(self.root)), env=env,
                                           stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                           shell=False, close_fds=True, pass_fds=fds, start_new_session=True)
                record['pid'] = process.pid
                self.persist('child_started', job=job['id'], kind=kind, seed=seed, pid=process.pid)
                while process.poll() is None:
                    # Even when stopping, finish/reap only OUR child, without signalling it.
                    self.sleep(self.plan['poll_seconds'])
                record.update(returncode=process.returncode, elapsed_seconds=max(0.000001, self.clock() - started),
                              ended_at=time.time())
            self.persist('child_exited', job=job['id'], kind=kind, seed=seed,
                         returncode=record['returncode'], elapsed_seconds=record['elapsed_seconds'])
            self.finish(job, kind, seed, record)
        except (OSError, ValueError, WorkerError) as exc:
            if process is not None:
                # A bookkeeping error must never let a second job overlap OUR child.
                while process.poll() is None:
                    self.sleep(self.plan['poll_seconds'])
            self.block(job, kind, seed, str(exc))

    def run(self):
        if sys.platform != 'linux' or fcntl is None:
            raise WorkerError('execution is Linux-server-only; --validate-plan is read-only')
        self.lane_root.mkdir(parents=True, exist_ok=True)
        gpu_lock_dir = inside(self.root / 'locks', self.root)
        gpu_lock_dir.mkdir(exist_ok=True)
        try:
            if not self.acquire(self.lane_root / 'worker.lock', True, report=False):
                return 130
            self.initialize_state()
            if not self.acquire(gpu_lock_dir / (self.plan['gpu_uuid'] + '.lock'), True):
                return 130
            if self.plan['legacy_lock_required']:
                if not self.acquire(Path(self.plan['legacy_lock']), False):
                    return 130
            self.reconcile()
            while not self.stop_requested:
                action = next_action(self.plan, self.state, self.probe_only)
                if action is None:
                    blocked = [j for j, v in self.state['jobs'].items() if v['status'] == 'BLOCKED']
                    if self.probe_only:
                        self.state['status'] = 'PROBES_COMPLETE_WITH_BLOCKERS' if blocked else 'PROBES_COMPLETE'
                        self.persist('probes_finished', blocked_jobs=blocked)
                        return 2 if blocked else 0
                    unfinished = [j for j, v in self.state['jobs'].items() if v['status'] not in {'BLOCKED', 'COMPLETE'}]
                    if unfinished:
                        self.state['status'] = 'BLOCKED_INVALID_STATE'
                        self.persist('no_runnable_action', unfinished=unfinished)
                        return 2
                    self.state['status'] = 'COMPLETE_WITH_BLOCKERS' if blocked else 'COMPLETE'
                    self.persist('lane_finished', blocked_jobs=blocked)
                    return 2 if blocked else 0
                job, kind, seed = action
                if not self.wait_ready(job):
                    break
                self.execute(job, kind, seed)
            self.state['status'] = 'STOPPED'
            self.persist('stopped_after_current_child')
            return 130
        except (OSError, ValueError, WorkerError) as exc:
            if self.state is not None:
                self.state['status'] = 'BLOCKED_WORKER'
                self.persist('worker_blocked', reason=str(exc))
            raise
        finally:
            for lock in reversed(self.locks):
                lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--lane')
    parser.add_argument('--validate-plan', action='store_true', help='schema validation only: no writes, GPU query or child process')
    parser.add_argument('--print-contract', action='store_true')
    parser.add_argument('--probe-only', action='store_true', help='run timing probes only; never formal training')
    args = parser.parse_args(argv)
    if args.print_contract:
        print(json.dumps(BASE_CONTRACT, indent=2))
        return 0
    if args.plan is None:
        parser.error('--plan is required')
    try:
        plan = validate_plan(read_json(args.plan), args.lane)
        if args.validate_plan:
            print(json.dumps({'status': 'PLAN_VALID', 'lane': plan['lane'], 'jobs': len(plan['jobs']),
                              'plan_sha256': canonical_sha(plan), 'gpu_accessed': False, 'files_written': False}))
            return 0
        if args.root is None or args.lane is None:
            parser.error('execution requires --root and --lane')
        worker = Worker(plan, args.root)
        worker.probe_only = args.probe_only
        signal.signal(signal.SIGINT, worker.stop)
        signal.signal(signal.SIGTERM, worker.stop)
        return worker.run()
    except (OSError, ValueError, WorkerError) as exc:
        print(json.dumps({'status': 'BLOCKED', 'reason': str(exc)}), file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    sys.exit(main())
