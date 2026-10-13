"""Copy small receipts and validate terminal results on the producing server."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import socket
import tarfile
import worker

ROOT = Path(__file__).resolve().parent
SMALL_FILES = {'state.json', 'progress.json', 'result.json', 'completion.json',
               'protocol.json', 'attempt.json', 'run_metadata.json', 'failure.json',
               'dependency_audit.json', 'test.json', 'source_audit.json',
               'readiness.json', 'audit.json', 'preparation_failed.json'}


def collect(snapshot):
    worker.safe_name(snapshot, 'snapshot')
    output = ROOT / 'evidence' / snapshot
    output.mkdir(parents=True, exist_ok=False)
    files, results, lanes = [], [], []
    for group in ('lanes', 'qa_smoke_01/lanes', 'plans', 'launch', 'dispatch',
                  'dependency_audit', 'taskdata', 'taskdata_native'):
        for path in sorted((ROOT / group).rglob('*')):
            if not path.is_file() or path.is_symlink():
                continue
            if path.name not in SMALL_FILES and not (
                    group in {'plans', 'launch', 'dispatch', 'dependency_audit'}
                    and path.suffix in {'.json', '.log'}):
                continue
            relative = path.relative_to(ROOT)
            data = path.read_bytes()
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            files.append({'path': str(relative), 'sha256': worker.file_sha(destination)})
    for state_path in sorted((output / 'lanes').glob('*/state.json')):
        state = worker.read_json(state_path)
        lane = state['lane']
        plan_path = output / 'plans' / (lane + '.production.json')
        plan = worker.validate_plan(worker.read_json(plan_path), lane)
        if worker.canonical_sha(plan) != state['plan_sha256']:
            raise RuntimeError('Frozen plan mismatch for ' + lane)
        jobs = {job['id']: job for job in plan['jobs']}
        lanes.append({'lane': lane, 'status': state['status'], 'updated_at': state['updated_at']})
        for identifier, current in state['jobs'].items():
            for seed, attempt in current['runs'].items():
                if attempt['status'] != 'COMPLETE':
                    continue
                directory = ROOT / 'lanes' / lane / 'jobs' / identifier / ('run' + seed)
                verified = worker.validate_training(directory, int(seed),
                    jobs[identifier].get('contract', {}), current['protocol_sha256'], jobs[identifier])
                payload = worker.read_json(directory / 'result.json')
                completion = worker.read_json(directory / 'completion.json')
                # External adapters additionally bind every listed output.
                for name, expected in completion.get('outputs', {}).items():
                    artifact = worker.inside(directory / name, directory)
                    if worker.file_sha(artifact) != expected:
                        raise RuntimeError('Completion output changed: ' + str(artifact))
                results.append({'lane': lane, 'id': identifier, 'seed': int(seed),
                    'dataset': jobs[identifier]['dataset'], 'method': jobs[identifier]['method'],
                    'test_evaluated': payload.get('test_evaluated'), 'verified': verified,
                    'epochs': payload.get('epochs_completed', payload.get('trained_epochs')),
                    'best_epoch': payload.get('best_epoch'),
                    'elapsed_seconds': payload.get('elapsed_seconds'),
                    'metric_unit': payload.get('metric_unit'),
                    'params': payload.get('params', payload.get('parameters'))})
    report = {'snapshot': snapshot, 'hostname': socket.gethostname(),
        'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'lanes': lanes, 'verified_complete_results': results, 'copied_files': files,
        'scope': 'Sequential snapshot; only terminal state-marked results verified; weights remain on server.'}
    worker.atomic_json(output / 'verification.json', report, overwrite=False)
    archive = output.with_suffix('.tar.gz')
    with tarfile.open(archive, 'x:gz') as package:
        package.add(output, arcname=snapshot)
    print(json.dumps({'archive': str(archive), 'sha256': worker.file_sha(archive),
                      'verified_results': len(results), 'lanes': lanes}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True)
    collect(parser.parse_args().snapshot)
