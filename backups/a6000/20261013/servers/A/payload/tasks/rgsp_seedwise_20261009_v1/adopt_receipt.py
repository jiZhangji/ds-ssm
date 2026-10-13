#!/usr/bin/env python3
"""Import a completed, uninterrupted legacy child after its safe handoff."""
import argparse
import json
import math
import shutil
from pathlib import Path
from manifest import digest, frozen, load_plan, read_json, sha, trial_by_id, verify_code


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    p.add_argument('--trial', required=True)
    p.add_argument('--legacy-completion', required=True)
    args = p.parse_args()
    plan = load_plan(args.plan)
    trial = trial_by_id(plan, args.trial)
    if trial['id'] != 'ssdd_local_energy_seed0' or trial['seed'] != 0:
        raise RuntimeError('Only the specifically adopted seed0 LocalEnergy is imported')
    verify_code(plan)
    finished = read_json(args.legacy_completion)
    if finished['status'] != 'adopted_and_scheduler_retired' or finished['seed'] != 0 or not finished['child_not_signaled']:
        raise RuntimeError('Legacy handoff is incomplete')
    origin = Path(finished['adopted_dir'])
    train = read_json(origin / 'train_complete.json')
    test = read_json(origin / 'eval_test.json')
    protocol = read_json(origin / 'protocol.json')
    if protocol['data_manifest_sha256'] != plan['datasets']['ssdd']['manifest_sha256']:
        raise RuntimeError('Adopted data split differs')
    audit = read_json(origin / 'source_training/runtime_audit.json')
    if audit.get('resumed') or (origin / 'source_training/runtime_audit_resume.json').exists():
        raise RuntimeError('Interrupted/resumed history is not a clean first-seed run')
    rows = [json.loads(line) for line in (origin / 'source_training/controls_by_epoch.jsonl').read_text().splitlines() if line.strip()]
    if [row['epoch'] for row in rows] != list(range(1, train['last_epoch_observed'] + 1)):
        raise RuntimeError('Adopted epoch audit has a gap')
    for row in rows:
        if any(not math.isfinite(v) for v in row['loss']):
            raise RuntimeError('Nonfinite legacy loss')
        if len(row['controls']) != 1 or row['controls'][0]['control'] != 'local_energy':
            raise RuntimeError('Adopted control differs')
    best_hash = sha(origin / 'weights/best.pt')
    if not (best_hash == train['best_sha256'] == test['weights_sha256'] == finished['best_sha256']):
        raise RuntimeError('Adoption hash chain differs')
    if test['split'] != 'test' or test['seed'] != 0 or train.get('smoke_only'):
        raise RuntimeError('Adopted result is not formal seed0 test')
    output = Path(trial['run_dir'])
    output.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ('train_complete.json', 'eval_test.json', 'weights/best.pt'):
        src, dst = origin / name, output / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open('rb') as incoming, dst.open('xb') as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        if sha(src) != sha(dst):
            raise RuntimeError('Adoption copy hash differs')
        copied.append({'path': str(dst), 'sha256': sha(dst), 'source': str(src)})
    frozen(output / 'adoption.json', {'status': 'verified', 'plan_sha256': digest(plan),
           'trial_sha256': digest(trial), 'origin': str(origin), 'artifacts': copied,
           'source_completion': {'path': args.legacy_completion, 'sha256': sha(args.legacy_completion)},
           'continuous_epochs_verified': len(rows),
           'limitations': 'Uninterrupted legacy-code training preserved; isolated frozen-code test after parent-only scheduler handoff. Not a newly trained run.'})
    frozen(output / 'completion.json', {'status': 'complete', 'plan_sha256': digest(plan),
           'trial_id': trial['id'], 'seed': 0, 'best_sha256': best_hash,
           'eval_test_sha256': sha(output / 'eval_test.json'), 'overall': test['overall']})
    print(json.dumps({'status': 'complete', 'trial_id': trial['id'], 'overall': test['overall']}))


if __name__ == '__main__':
    main()
