"""Detached, finite predecessor gate for a fresh lane; never replace attempts."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import worker
from launch_lane import identity

ROOT = Path(__file__).resolve().parent
TERMINAL = {'COMPLETE', 'COMPLETE_WITH_BLOCKERS'}


def controller(lane, predecessor):
    directory = ROOT / 'dispatch'
    with (directory / (lane + '.gate.lock')).open('ab') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + 14 * 86400
        while True:
            if time.monotonic() > deadline:
                raise RuntimeError('Finite predecessor wait expired')
            state_path = ROOT / 'lanes' / predecessor / 'state.json'
            if not state_path.exists():
                worker.atomic_json(directory / (lane + '.gate.state.json'),
                    {'state': 'WAITING_PREDECESSOR_START', 'predecessor': predecessor, 'time': time.time()})
                time.sleep(10)
                continue
            state = worker.read_json(state_path)
            receipt_path = ROOT / 'launch' / (predecessor + '.json')
            receipt = worker.read_json(receipt_path)
            live = identity(receipt['pid'])
            running = bool(live and live['start_ticks'] == receipt['start_ticks'])
            if state['status'] in TERMINAL and not running:
                break
            if state['status'] in {'STOPPED', 'BLOCKED_WORKER', 'BLOCKED_INVALID_STATE'}:
                raise RuntimeError('Predecessor requires review: ' + state['status'])
            worker.atomic_json(directory / (lane + '.gate.state.json'),
                {'state': 'WAITING_PREDECESSOR', 'predecessor': predecessor, 'time': time.time()})
            time.sleep(10)
        subprocess.run([sys.executable, '-B', str(ROOT / 'launch_lane.py'),
                        '--lane', lane, '--probe-only'], check=True, cwd=ROOT)
        subprocess.run([sys.executable, '-B', str(ROOT / 'after_preflight.py'),
                        '--lane', lane], check=True, cwd=ROOT)
        worker.atomic_json(directory / (lane + '.gate.state.json'),
            {'state': 'PREFLIGHT_DISPATCHED', 'predecessor': predecessor,
             'predecessor_state_sha256': worker.file_sha(state_path), 'time': time.time()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lane', required=True)
    parser.add_argument('--predecessor', required=True)
    parser.add_argument('--foreground', action='store_true')
    args = parser.parse_args()
    for name in (args.lane, args.predecessor):
        worker.safe_name(name, 'lane')
    if args.lane == args.predecessor:
        raise ValueError('A lane cannot follow itself')
    worker.validate_plan(worker.read_json(ROOT / 'plans' / (args.lane + '.json')), args.lane)
    worker.validate_plan(worker.read_json(ROOT / 'plans' / (args.predecessor + '.json')), args.predecessor)
    directory = ROOT / 'dispatch'
    directory.mkdir(exist_ok=True)
    if args.foreground:
        try:
            return controller(args.lane, args.predecessor)
        except Exception as error:
            worker.atomic_json(directory / (args.lane + '.gate.state.json'),
                {'state': 'FAILED', 'predecessor': args.predecessor,
                 'error': str(error), 'time': time.time()})
            raise
    with (directory / (args.lane + '.gate.launch.lock')).open('ab') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = directory / (args.lane + '.gate.launch.json')
        if receipt_path.exists():
            prior = worker.read_json(receipt_path)
            if prior['predecessor'] != args.predecessor:
                raise RuntimeError('Prior predecessor differs')
            print(json.dumps({'status': 'PRIOR_GATE_PRESERVED', **prior}))
            return
        command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()),
                   '--lane', args.lane, '--predecessor', args.predecessor, '--foreground']
        with (directory / (args.lane + '.gate.log')).open('xb') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        live = identity(child.pid)
        if not live:
            raise RuntimeError('Gate exited before receipt')
        receipt = {'lane': args.lane, 'predecessor': args.predecessor,
                   'pid': child.pid, 'start_ticks': live['start_ticks'], 'command': command}
        worker.atomic_json(receipt_path, receipt, overwrite=False)
        print(json.dumps({'status': 'GATE_DISPATCHED', **receipt}))


if __name__ == '__main__':
    main()
