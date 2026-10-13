"""One-shot, user-authorized priority-only revision of the stopped B lane."""
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import worker
from launch_lane import identity

ROOT = Path(__file__).resolve().parent


def main():
    directory = ROOT / 'reprioritize_ours_01'
    receipt = worker.read_json(ROOT / 'launch/b0acd.json')
    live = identity(receipt['pid'])
    assert not live or live['start_ticks'] != receipt['start_ticks'], 'Wait for the current child to finish'
    path = ROOT / 'plans/b0acd.production.json'
    lane = ROOT / 'lanes/b0acd'
    with (lane / 'worker.lock').open('ab') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = worker.read_json(lane / 'state.json')
        assert state['status'] == 'STOPPED'
        assert all(v['status'] != 'RUNNING' for j in state['jobs'].values() for v in [j['probe'], *j['runs'].values()])
        original = worker.validate_plan(worker.read_json(path), 'b0acd')
        assert state['plan_sha256'] == worker.canonical_sha(original)
        revised = copy.deepcopy(original)
        for job in revised['jobs']:
            job['stage'] = 0 if job['method'] in {'Original', 'Matched075', 'Full'} else 20
        revised = worker.validate_plan(revised, 'b0acd')
        for before, after in zip(original['jobs'], revised['jobs']):
            assert {k:v for k,v in before.items() if k != 'stage'} == {k:v for k,v in after.items() if k != 'stage'}
        worker.atomic_json(directory / 'prior_plan.json', original, overwrite=False)
        worker.atomic_json(directory / 'prior_state.json', state, overwrite=False)
        worker.atomic_json(directory / 'prior_launch.json', receipt, overwrite=False)
        worker.atomic_json(path, revised)
        state['plan_sha256'] = worker.canonical_sha(revised)
        state['revision'] += 1
        state['status'] = 'READY'
        worker.atomic_json(lane / 'state.json', state)
        worker.atomic_json(directory / 'revision.json', {'status':'PRIORITY_ONLY_REVISED',
            'before_sha256': worker.canonical_sha(original), 'after_sha256': worker.canonical_sha(revised),
            'training_configuration_changed': False, 'prior_attempts_preserved': True}, overwrite=False)
    command = [sys.executable, '-B', '-u', str(ROOT / 'worker.py'), '--root', str(ROOT),
               '--plan', str(path), '--lane', 'b0acd']
    with (directory / 'worker.log').open('xb') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    live = identity(child.pid)
    assert live is not None
    result = {'lane': 'b0acd', 'pid': child.pid, 'start_ticks': live['start_ticks'],
              'command': command, 'plan_sha256': worker.canonical_sha(revised)}
    worker.atomic_json(directory / 'launch.json', result, overwrite=False)
    worker.atomic_json(ROOT / 'launch/b0acd.json', result)
    print(json.dumps({'status': 'RESUMED_OUR_METHODS_FIRST', **result}))


if __name__ == '__main__':
    main()
