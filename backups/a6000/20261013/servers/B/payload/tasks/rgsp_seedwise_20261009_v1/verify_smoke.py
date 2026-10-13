#!/usr/bin/env python3
"""Verify finished smoke evidence and preserve its exact source before freeze."""
import argparse
import json
import math
import tarfile
from pathlib import Path

from manifest import (digest, frozen, load_plan, read_json, receipt_path, sha,
                      trial_by_id, utc, validate_receipt, verify_code)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--lane', required=True)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if plan.get('smoke_only') is not True:
        raise RuntimeError('This verifier only accepts independent smoke plans')
    verify_code(plan)
    lane = next(item for item in plan['lanes'] if item['id'] == args.lane)
    completion_path = Path(lane['output_dir']) / 'lane_complete.json'
    completion = read_json(completion_path)
    if (completion['plan_sha256'] != digest(plan)
            or completion['completed'] != lane['trial_ids']
            or completion['trial_ids'] != lane['trial_ids']):
        raise RuntimeError('Incomplete smoke lane')
    evidence = []
    for tid in lane['trial_ids']:
        trial = trial_by_id(plan, tid)
        receipt = validate_receipt(plan, trial, read_json(receipt_path(plan, trial)), verify_files=True)
        out = Path(trial['run_dir'])
        train = read_json(out / 'train_complete.json')
        test = read_json(out / 'eval_test.json')
        rows = [json.loads(line) for line in (out / 'controls_by_epoch.jsonl').read_text().splitlines() if line.strip()]
        if train['last_epoch_observed'] != 2 or train.get('smoke_only') is not True:
            raise RuntimeError('Wrong smoke epoch budget')
        if [row['epoch'] for row in rows] != [1, 2] or not all(
                math.isfinite(value) for row in rows for value in row['loss']):
            raise RuntimeError('Missing epoch or nonfinite smoke loss')
        if not (sha(out / 'weights/best.pt') == train['best_sha256'] == test['weights_sha256']):
            raise RuntimeError('Smoke checkpoint/evaluation hash chain differs')
        if test['split'] != 'test' or not all(math.isfinite(v) for v in test['overall'].values()):
            raise RuntimeError('Smoke test is not finite')
        evidence.append({'trial_id': tid, 'epochs': 2, 'losses': [row['loss'] for row in rows],
                         'receipt': receipt, 'test_overall': test['overall']})
    task = Path(plan['source_root']).parent
    archive = task / 'evidence' / ('preformal_smoke_' + args.lane + '_source.tar.gz')
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        with tarfile.open(str(archive), 'x:gz') as bundle:
            for item in plan['code_hashes']:
                path = Path(item['path'])
                bundle.add(str(path), arcname=str(path.relative_to(task)), recursive=False)
    record = {'status': 'verified', 'verified_utc': utc(), 'plan_sha256': digest(plan),
              'lane': args.lane, 'trials': evidence, 'formal_result': False,
              'source_snapshot': {'path': str(archive), 'sha256': sha(archive)},
              'lane_completion': {'path': str(completion_path), 'sha256': sha(completion_path)}}
    target = task / 'evidence' / ('smoke_' + args.lane + '_verified.json')
    frozen(target, record)
    print(json.dumps({'status': 'verified', 'trials': [item['trial_id'] for item in evidence],
                      'evidence': str(target), 'sha256': sha(target),
                      'source_snapshot': record['source_snapshot']}, sort_keys=True))


if __name__ == '__main__':
    main()
