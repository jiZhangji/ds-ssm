#!/usr/bin/env python3
"""Launch this task's lane once; preserve failed or uncertain attempts for review."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import worker

ROOT = Path(__file__).resolve().parent


def identity(pid):
    """Return Linux PID/start-time identity; access errors never mean 'exited'."""
    if type(pid) is not int or pid <= 0:
        raise worker.WorkerError('owner PID must be a positive integer')
    directory = Path('/proc') / str(pid)
    try:
        fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        argv = (directory / 'cmdline').read_bytes().split(b'\0')
        second = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    if len(fields) < 20 or len(second) < 20 or fields[19] != second[19]:
        raise worker.WorkerError('process identity changed while reading /proc')
    return {'pid': pid, 'start_ticks': fields[19], 'process_state': second[0],
            'argv': [arg.decode('utf-8', errors='replace') for arg in argv if arg]}


def owner_running(receipt):
    ticks = receipt.get('start_ticks')
    if not isinstance(ticks, str) or not ticks.isdigit():
        raise worker.WorkerError('owner receipt has no valid start_ticks')
    live = identity(receipt.get('pid'))
    return bool(live and live['start_ticks'] == ticks and live['process_state'] not in {'Z', 'X'})


def existing_attempt(receipt_path, lane, digest):
    previous = worker.read_json(receipt_path)
    if previous.get('lane') != lane or previous.get('plan_sha256') != digest:
        raise worker.WorkerError('prior launch belongs to another lane or plan')
    if previous.get('root') != str(ROOT):
        raise worker.WorkerError('prior launch belongs to another task root')
    if owner_running(previous):
        return {'status': 'ALREADY_RUNNING', **previous}
    return {'status': 'PRIOR_ATTEMPT_PRESERVED', 'lane': lane,
            'state': str(ROOT / 'lanes' / lane / 'state.json'), 'receipt': str(receipt_path)}


def launch(lane):
    if sys.platform != 'linux' or worker.fcntl is None:
        raise worker.WorkerError('launch is Linux-server-only')
    worker.safe_name(lane, 'lane')
    plan_path = ROOT / 'plans' / (lane + '.production.json')
    plan = worker.validate_plan(worker.read_json(plan_path), lane)
    digest = worker.canonical_sha(plan)
    directory = ROOT / 'launch'
    directory.mkdir(exist_ok=True)
    with (directory / (lane + '.lock')).open('ab') as lock:
        worker.fcntl.flock(lock, worker.fcntl.LOCK_EX | worker.fcntl.LOCK_NB)
        receipt_path = directory / (lane + '.json')
        if receipt_path.exists():
            return existing_attempt(receipt_path, lane, digest)
        intent_path = directory / (lane + '.intent.json')
        log_path = directory / (lane + '.log')
        if intent_path.exists() or log_path.exists() or (ROOT / 'lanes' / lane).exists():
            raise worker.WorkerError('earlier launch files exist without a receipt; preserve for review')
        command = [sys.executable, '-B', '-u', str(ROOT / 'worker.py'), '--root', str(ROOT),
                   '--plan', str(plan_path), '--lane', lane]
        intent = {'lane': lane, 'root': str(ROOT), 'command': command,
                  'plan_sha256': digest, 'created_at': time.time()}
        worker.atomic_json(intent_path, intent, overwrite=False)
        # Reserving the log before fork prevents an unreceipted child being retried.
        with log_path.open('xb') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1'))
        live = identity(child.pid)
        if live is None or live['process_state'] in {'Z', 'X'}:
            raise worker.WorkerError('worker exited before launch identity could be recorded')
        record = {**intent, 'pid': child.pid, 'start_ticks': live['start_ticks'],
                  'log_path': str(log_path), 'launched_at': time.time()}
        worker.atomic_json(receipt_path, record, overwrite=False)
        return {'status': 'STARTED', **record}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane', required=True)
    args = parser.parse_args(argv)
    try:
        result = launch(args.lane)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result['status'] in {'STARTED', 'ALREADY_RUNNING'} else 2
    except (OSError, ValueError, worker.WorkerError) as error:
        print(json.dumps({'status': 'REQUIRES_REVIEW', 'lane': args.lane,
                          'error': str(error)}), flush=True)
        return 2


if __name__ == '__main__':
    sys.exit(main())
