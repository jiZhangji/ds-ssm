#!/usr/bin/env python3
"""Finite SAMPLE gate: predecessor terminates, data is ready, build plan, launch.

Only this independent task is written. The predecessor is read-only. A missing
predecessor state/launch is allowed while its own upstream gate is waiting.
Stopped, failed, corrupt, or uncertain work requires review and is never retried.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import worker
from launch_lane import identity, owner_running

ROOT = Path(__file__).resolve().parent
WAIT_SECONDS = 14 * 86400
POLL_SECONDS = 15
SUCCESS = {'COMPLETE', 'COMPLETE_WITH_BLOCKERS'}
REQUIRED_RUNTIME = ('worker.py', 'launch_lane.py', 'queue_after.py',
                    'build_plan.py', 'run_amplitude.py', 'data/prepare_data.py')


def require_runtime():
    missing = [name for name in REQUIRED_RUNTIME if not (ROOT / name).is_file()]
    if missing:
        raise worker.WorkerError('required task runtime is missing: ' + ', '.join(missing))


def validate_args(args):
    if sys.platform != 'linux' or worker.fcntl is None:
        raise worker.WorkerError('gate execution is Linux-server-only')
    worker.safe_name(args.lane, 'lane')
    worker.safe_name(args.predecessor_lane, 'predecessor lane')
    if not args.predecessor_root.is_absolute():
        raise worker.WorkerError('--predecessor-root must be absolute')
    args.predecessor_root = args.predecessor_root.resolve()
    if args.predecessor_root == ROOT or ROOT in args.predecessor_root.parents:
        raise worker.WorkerError('predecessor must be another independent task')
    if not re.fullmatch(r'GPU-[A-Za-z0-9-]+', args.gpu_uuid):
        raise worker.WorkerError('--gpu-uuid must be a full GPU UUID')
    require_runtime()


def binding(args):
    return {'root': str(ROOT), 'lane': args.lane, 'gpu_uuid': args.gpu_uuid,
            'predecessor_root': str(args.predecessor_root),
            'predecessor_lane': args.predecessor_lane, 'methods':args.methods}


def read_bound_state(path, lane, root):
    state = worker.read_json(path)
    if state.get('lane') != lane:
        raise worker.WorkerError('predecessor state lane identity mismatch')
    if state.get('root') != str(root):
        raise worker.WorkerError('predecessor state root identity mismatch')
    if not isinstance(state.get('status'), str):
        raise worker.WorkerError('predecessor state has no status')
    return state


def predecessor_status(args):
    state_path = args.predecessor_root / 'lanes' / args.predecessor_lane / 'state.json'
    receipt_path = args.predecessor_root / 'launch' / (args.predecessor_lane + '.json')
    detail = {'state_path': str(state_path), 'receipt_path': str(receipt_path)}
    if not state_path.exists() and not receipt_path.exists():
        upstream_path = args.predecessor_root / 'dispatch' / (args.predecessor_lane + '.gate.state.json')
        if upstream_path.exists():
            upstream = worker.read_json(upstream_path)
            upstream_status = upstream.get('status')
            if not isinstance(upstream_status, str):
                raise worker.WorkerError('predecessor gate state has no status')
            detail.update(upstream_gate_state=str(upstream_path), upstream_gate_status=upstream_status)
            if upstream_status.startswith(('STOPPED', 'BLOCKED', 'FAILED', 'REQUIRES_REVIEW')):
                raise worker.WorkerError('predecessor gate requires review: ' + upstream_status)
        return False, 'WAITING_PREDECESSOR_NOT_STARTED', detail
    receipt = worker.read_json(receipt_path) if receipt_path.exists() else None
    if receipt is not None and receipt.get('lane') != args.predecessor_lane:
        raise worker.WorkerError('predecessor launch lane identity mismatch')
    if receipt is not None:
        if 'root' in receipt and receipt['root'] != str(args.predecessor_root):
            raise worker.WorkerError('predecessor launch root identity mismatch')
        command = receipt.get('command')
        if not isinstance(command, list) or not all(isinstance(arg, str) for arg in command):
            raise worker.WorkerError('predecessor launch command is invalid')
        for flag, expected in (('--root', str(args.predecessor_root)),
                               ('--lane', args.predecessor_lane)):
            if command.count(flag) != 1 or command.index(flag) + 1 >= len(command) or command[command.index(flag) + 1] != expected:
                raise worker.WorkerError('predecessor launch command binding mismatch: ' + flag)
        if str(args.predecessor_root / 'worker.py') not in command:
            raise worker.WorkerError('predecessor launch must name its own worker')
    running = owner_running(receipt) if receipt is not None else None
    if not state_path.exists():
        if not running:
            raise worker.WorkerError('predecessor owner exited without a state file')
        return False, 'WAITING_PREDECESSOR_INITIAL_STATE', detail
    state = read_bound_state(state_path, args.predecessor_lane, args.predecessor_root)
    if state.get('gpu_uuid') != args.gpu_uuid:
        raise worker.WorkerError('predecessor is assigned to another GPU')
    status = state['status']
    detail.update(predecessor_status=status, owner_running=running)
    if status == 'STOPPED' or status.startswith(('BLOCKED', 'FAILED', 'REQUIRES_REVIEW')):
        raise worker.WorkerError('predecessor requires review: ' + status)
    if receipt is None:
        if status in SUCCESS:
            raise worker.WorkerError('terminal predecessor has no owner launch receipt')
        return False, 'WAITING_PREDECESSOR_LAUNCH_RECEIPT', detail
    if worker.valid_sha(state.get('plan_sha256'), 'predecessor state plan') != worker.valid_sha(
            receipt.get('plan_sha256'), 'predecessor launch plan'):
        raise worker.WorkerError('predecessor state and launch plan hashes disagree')
    if status in SUCCESS:
        if running:
            return False, 'WAITING_PREDECESSOR_OWNER_EXIT', detail
        detail.update(state_sha256=worker.file_sha(state_path),
                      launch_sha256=worker.file_sha(receipt_path),
                      owner_pid=receipt['pid'], owner_start_ticks=receipt['start_ticks'])
        return True, 'PREDECESSOR_COMPLETE', detail
    if not running:
        raise worker.WorkerError('predecessor owner exited before a successful terminal state: ' + status)
    return False, 'WAITING_PREDECESSOR', detail


def data_status():
    ready_path = ROOT / 'data' / 'READY.json'
    detail = {'ready_path': str(ready_path)}
    failure_path = ROOT / 'data' / 'FAILED.json'
    if failure_path.exists():
        raise worker.WorkerError('data preparation has an unresolved failure: ' + str(failure_path))
    if not ready_path.exists():
        return False, 'WAITING_DATA', detail
    ready = worker.read_json(ready_path)
    if ready.get('status') != 'READY':
        raise worker.WorkerError('data READY receipt is not successful: ' + str(ready.get('status')))
    if ready.get('schema') != 'sample-amplitude-ready-v1' or ready.get('data_root') != str(ROOT / 'data'):
        raise worker.WorkerError('data READY identity does not match this amplitude task')
    if any(ready.get(key) != value for key, value in (('count', 1345), ('train_count', 806), ('test_count', 539))):
        raise worker.WorkerError('data READY sample counts do not match the frozen protocol')
    for split in ('train', 'test'):
        worker.valid_sha(ready.get('manifest_' + split + '_sha256'), split + ' manifest hash')
    # The builder and runner verify every manifest/content binding before use.
    detail.update(ready_sha256=worker.file_sha(ready_path), data_status=ready['status'])
    return True, 'DATA_READY', detail


class Gate:
    def __init__(self, args):
        self.args = args
        self.directory = ROOT / 'dispatch'
        self.state_path = self.directory / (args.lane + '.gate.state.json')
        self.stop_requested = False
        self.last_logged_status = None

    def stop(self, *_):
        self.stop_requested = True

    def persist(self, status, **detail):
        record = {'status': status, **binding(self.args), 'time': time.time(), **detail}
        worker.atomic_json(self.state_path, record)
        if status != self.last_logged_status:
            print(json.dumps(record, sort_keys=True), flush=True)
            self.last_logged_status = status

    def build_and_launch(self, predecessor, data):
        # A unique invocation log prevents automatic replay of a partial builder.
        plan_path = ROOT / 'plans' / (self.args.lane + '.production.json')
        build_log = self.directory / (self.args.lane + '.build_plan.log')
        command = [sys.executable, '-B', str(ROOT / 'build_plan.py'), '--root', str(ROOT),
                   '--lane', self.args.lane, '--gpu-uuid', self.args.gpu_uuid,
                   '--methods', *self.args.methods]
        self.persist('BUILDING_PLAN', predecessor=predecessor, data=data, command=command)
        with build_log.open('xb') as log:
            built = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                                   timeout=600, check=False)
        if built.returncode:
            raise worker.WorkerError('build_plan failed; inspect ' + str(build_log))
        plan = worker.validate_plan(worker.read_json(plan_path), self.args.lane)
        if plan['gpu_uuid'] != self.args.gpu_uuid:
            raise worker.WorkerError('built plan GPU does not match the queued GPU')
        # Recheck the read-only gates immediately before starting the new owner.
        predecessor_ok, _, predecessor_now = predecessor_status(self.args)
        data_ok, _, data_now = data_status()
        if not predecessor_ok or not data_ok or data_now['ready_sha256'] != data['ready_sha256']:
            raise worker.WorkerError('gate prerequisites changed during plan creation')
        if any(predecessor_now[field] != predecessor[field] for field in ('launch_sha256', 'state_sha256')):
            raise worker.WorkerError('predecessor terminal evidence changed during plan creation')
        if self.stop_requested:
            raise worker.WorkerError('gate stopped before dispatch')
        command = [sys.executable, '-B', str(ROOT / 'launch_lane.py'), '--lane', self.args.lane]
        self.persist('LAUNCHING', predecessor=predecessor_now, data=data_now,
                     plan_sha256=worker.canonical_sha(plan))
        result = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                text=True, capture_output=True, timeout=60, check=False)
        result_path = self.directory / (self.args.lane + '.launch_result.json')
        worker.atomic_json(result_path, {'returncode': result.returncode,
                           'stdout': result.stdout, 'stderr': result.stderr}, overwrite=False)
        if result.returncode:
            raise worker.WorkerError('lane launcher failed; inspect ' + str(result_path))
        response = json.loads(result.stdout)
        if response.get('status') not in {'STARTED', 'ALREADY_RUNNING'}:
            raise worker.WorkerError('launcher did not confirm an active new owner')
        receipt_path = ROOT / 'launch' / (self.args.lane + '.json')
        receipt = worker.read_json(receipt_path)
        if receipt.get('lane') != self.args.lane or receipt.get('plan_sha256') != worker.canonical_sha(plan):
            raise worker.WorkerError('new owner launch receipt identity mismatch')
        if not owner_running(receipt):
            raise worker.WorkerError('new worker exited immediately after launch; inspect lane/log')
        self.persist('DISPATCHED', predecessor=predecessor_now, data=data_now,
                     plan_sha256=worker.canonical_sha(plan), result=response,
                     launch_sha256=worker.file_sha(receipt_path))

    def run(self):
        # Only the lock owner may write gate state, including failure state.
        with (self.directory / (self.args.lane + '.gate.lock')).open('ab') as lock:
            worker.fcntl.flock(lock, worker.fcntl.LOCK_EX | worker.fcntl.LOCK_NB)
            if self.state_path.exists():
                previous = worker.read_json(self.state_path)
                raise worker.WorkerError('prior gate state is preserved; no automatic restart: '
                                         + str(previous.get('status')))
            deadline = time.monotonic() + WAIT_SECONDS
            self.persist('QUEUED', expires_at=time.time() + WAIT_SECONDS)
            try:
                while time.monotonic() < deadline:
                    if self.stop_requested:
                        raise worker.WorkerError('gate stop requested; predecessor was not signalled')
                    ready, ready_status, data = data_status()
                    complete, predecessor_label, predecessor = predecessor_status(self.args)
                    if complete and ready:
                        self.build_and_launch(predecessor, data)
                        return 0
                    status = predecessor_label if not complete else ready_status
                    self.persist(status, predecessor=predecessor, data=data,
                                 data_ready=ready, predecessor_complete=complete)
                    time.sleep(POLL_SECONDS)
                raise worker.WorkerError('14-day prerequisite wait expired; no lane was launched')
            except Exception as error:
                self.persist('REQUIRES_REVIEW', error=str(error))
                raise


def enqueue(args):
    directory = ROOT / 'dispatch'
    directory.mkdir(exist_ok=True)
    with (directory / (args.lane + '.gate.launch.lock')).open('ab') as lock:
        worker.fcntl.flock(lock, worker.fcntl.LOCK_EX | worker.fcntl.LOCK_NB)
        receipt_path = directory / (args.lane + '.gate.launch.json')
        if receipt_path.exists():
            previous = worker.read_json(receipt_path)
            if any(previous.get(key) != value for key, value in binding(args).items()):
                raise worker.WorkerError('prior gate is bound to different prerequisites')
            status = 'ALREADY_QUEUED' if owner_running(previous) else 'PRIOR_GATE_PRESERVED'
            return {'status': status, **previous}
        intent_path = directory / (args.lane + '.gate.intent.json')
        log_path = directory / (args.lane + '.gate.log')
        state_path = directory / (args.lane + '.gate.state.json')
        if any(path.exists() for path in (intent_path, log_path, state_path)):
            raise worker.WorkerError('prior gate artifacts lack a launch receipt; preserve for review')
        command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--foreground',
                   '--predecessor-root', str(args.predecessor_root),
                   '--predecessor-lane', args.predecessor_lane, '--lane', args.lane,
                   '--gpu-uuid', args.gpu_uuid, '--methods', *args.methods]
        intent = {**binding(args), 'command': command, 'created_at': time.time()}
        worker.atomic_json(intent_path, intent, overwrite=False)
        with log_path.open('xb') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1'))
        live = identity(child.pid)
        if live is None or live['process_state'] in {'Z', 'X'}:
            raise worker.WorkerError('gate exited before its owner identity was recorded')
        receipt = {**intent, 'pid': child.pid, 'start_ticks': live['start_ticks'],
                   'log_path': str(log_path), 'launched_at': time.time()}
        worker.atomic_json(receipt_path, receipt, overwrite=False)
        return {'status': 'QUEUED', **receipt}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predecessor-root', type=Path, required=True)
    parser.add_argument('--predecessor-lane', required=True)
    parser.add_argument('--lane', default='a0amplitude')
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--methods',nargs='+',choices=('Full','Original','Matched075','ResNet50'),
                        default=['Full','Original','Matched075','ResNet50'])
    parser.add_argument('--foreground', action='store_true')
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        (ROOT / 'dispatch').mkdir(exist_ok=True)
        if args.foreground:
            gate = Gate(args)
            signal.signal(signal.SIGINT, gate.stop)
            signal.signal(signal.SIGTERM, gate.stop)
            return gate.run()
        result = enqueue(args)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result['status'] in {'QUEUED', 'ALREADY_QUEUED'} else 2
    except (OSError, ValueError, worker.WorkerError, subprocess.SubprocessError) as error:
        # A process that failed to acquire the gate lock must not overwrite state.
        print(json.dumps({'status': 'REQUIRES_REVIEW', 'error': str(error)}), flush=True)
        return 2


if __name__ == '__main__':
    sys.exit(main())
