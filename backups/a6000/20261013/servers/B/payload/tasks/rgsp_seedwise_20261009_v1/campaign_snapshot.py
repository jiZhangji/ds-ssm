#!/usr/bin/env python3
"""Small read-only live snapshot; receipt presence is not completion verification."""
import argparse
import json
import math
import os
import shutil
import socket
from pathlib import Path

from campaign_dispatch import identity_alive
from manifest import digest, frozen, load_plan, read_json, sha, utc
from run_lane import gpu_snapshot


def optional_json(path):
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except (ValueError, OSError) as error:
        return {'read_error': str(error)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    plan = load_plan(args.plan)
    root = Path(plan['campaign_root'])
    hostname = socket.gethostname()
    keys = [key for key, item in plan['dispatch_map'].items() if item['hostname'] == hostname]
    if not keys:
        raise RuntimeError('This host is not assigned by the frozen plan')
    result = {'schema': 'rgsp-live-snapshot-v1', 'checked_utc': utc(), 'hostname': hostname,
              'plan_sha256': digest(plan), 'plan_file_sha256': sha(args.plan),
              'disk_free_gib': round(shutil.disk_usage(root).free / 1024**3, 2),
              'cpu_load': list(os.getloadavg()), 'hosts': {},
              'warning': 'Receipt presence and last epoch are progress only; verify_receipts is required for completion.'}
    for key in keys:
        entry = plan['dispatch_map'][key]
        dispatch = optional_json(root / 'dispatch' / key / 'status.json')
        item = {'dispatch': dispatch, 'gpu': gpu_snapshot(entry['gpu_uuid']), 'lanes': {}}
        if dispatch and dispatch.get('owner'):
            item['dispatcher_alive'] = identity_alive(dispatch['owner'])
        for seed in (0, 1):
            lane = next(row for row in plan['lanes'] if row['id'] == entry[f'seed{seed}_lane'])
            status = optional_json(Path(lane['output_dir']) / 'status.json')
            present = [tid for tid in lane['trial_ids'] if (Path(plan['receipts_dir']) / (tid + '.json')).is_file()]
            progress = {'status': status, 'assigned_count': len(lane['trial_ids']),
                        'receipt_files_present': present}
            if status and status.get('current'):
                trial = next(row for row in plan['trials'] if row['id'] == status['current'])
                out = Path(trial['run_dir'])
                epochs = out / 'controls_by_epoch.jsonl'
                if epochs.exists():
                    lines = epochs.read_text().splitlines()
                    if lines:
                        try:
                            row = json.loads(lines[-1])
                            progress['last_completed_epoch'] = row['epoch']
                            progress['last_loss'] = row['loss']
                            progress['last_loss_finite'] = all(math.isfinite(v) for v in row['loss'])
                        except (ValueError, KeyError):
                            progress['epoch_read_pending'] = True
                cross_last = optional_json(out / 'status.json')
                if cross_last is not None:
                    progress['cross_status'] = cross_last
                if trial.get('runner_kind') == 'deferred_detection':
                    data = plan['datasets'][trial['dataset']]
                    ready = Path(data.get('readiness', str(Path(data['descriptor']).parent / 'readiness.json')))
                    progress['deferred_readiness'] = optional_json(ready) or {'status': 'not_published'}
                if status.get('state') == 'failed':
                    log = Path(lane['output_dir']) / 'logs' / (trial['id'] + '.log')
                    if log.exists():
                        with log.open('rb') as stream:
                            stream.seek(max(0, log.stat().st_size - 4096))
                            progress['failure_log_tail'] = stream.read().decode(errors='replace')[-3000:]
            item['lanes'][lane['id']] = progress
        result['hosts'][key] = item
    if args.output:
        destination = Path(args.output).resolve()
        evidence = (root / 'evidence').resolve()
        if evidence not in destination.parents:
            raise RuntimeError('Snapshot output must be a new file inside task/evidence')
        frozen(destination, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
