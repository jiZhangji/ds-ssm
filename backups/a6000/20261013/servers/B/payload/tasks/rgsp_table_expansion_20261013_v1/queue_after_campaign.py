"""Finite A-lane gate after the registered prior campaign; no preemption."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import worker
from launch_lane import identity

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'rgsp_unified_a6000_20261012_v2'
LANE, PREDECESSOR = 'a0acd', 'a0samplex'


def gate():
    directory = ROOT / 'dispatch'
    with (directory / (LANE + '.gate.lock')).open('ab') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + 14 * 86400
        while time.monotonic() < deadline:
            state_path = PREVIOUS / 'lanes' / PREDECESSOR / 'state.json'
            state = worker.read_json(state_path)
            receipt = worker.read_json(PREVIOUS / 'launch' / (PREDECESSOR + '.json'))
            live = identity(receipt['pid'])
            running = live and live['start_ticks'] == receipt['start_ticks']
            if state['status'] in {'COMPLETE', 'COMPLETE_WITH_BLOCKERS'} and not running:
                result = subprocess.run([sys.executable, '-B', str(ROOT / 'launch_lane.py'), '--lane', LANE],
                    cwd=ROOT, text=True, capture_output=True, check=True)
                worker.atomic_json(directory / (LANE + '.gate.state.json'),
                    {'status': 'DISPATCHED', 'predecessor_state_sha256': worker.file_sha(state_path),
                     'result': json.loads(result.stdout), 'time': time.time()})
                return
            if state['status'] in {'STOPPED', 'BLOCKED_WORKER', 'BLOCKED_INVALID_STATE'}:
                raise RuntimeError('Predecessor requires review: ' + state['status'])
            worker.atomic_json(directory / (LANE + '.gate.state.json'),
                {'status': 'WAITING_PREDECESSOR', 'predecessor': str(state_path), 'time': time.time()})
            time.sleep(15)
        raise RuntimeError('14-day predecessor wait expired')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--foreground', action='store_true')
    args = parser.parse_args()
    worker.validate_plan(worker.read_json(ROOT / 'plans' / (LANE + '.production.json')), LANE)
    directory = ROOT / 'dispatch'
    directory.mkdir(exist_ok=True)
    if args.foreground:
        try:
            gate()
        except Exception as error:
            worker.atomic_json(directory / (LANE + '.gate.state.json'), {'status': 'FAILED', 'error': str(error), 'time': time.time()})
            raise
        return
    with (directory / (LANE + '.gate.launch.lock')).open('ab') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = directory / (LANE + '.gate.launch.json')
        if receipt_path.exists():
            print(json.dumps({'status': 'PRIOR_GATE_PRESERVED', **worker.read_json(receipt_path)}))
            return
        command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--foreground']
        with (directory / (LANE + '.gate.log')).open('xb') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        live = identity(child.pid)
        if not live:
            raise RuntimeError('Gate exited before recording identity')
        receipt = {'pid': child.pid, 'start_ticks': live['start_ticks'], 'command': command,
                   'predecessor': str(PREVIOUS / 'lanes' / PREDECESSOR)}
        worker.atomic_json(receipt_path, receipt, overwrite=False)
        print(json.dumps({'status': 'QUEUED', **receipt}))


if __name__ == '__main__':
    main()
