#!/usr/bin/env python3
"""Finite SADD cross-dataset controls. Never changes the SSDD campaign."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

import run_rgsp_controls as core
from audit_sadd_data import audit

ROOT = core.ROOT
VARIANTS = ("original", "matched_scale075", "local_energy", "full")
CAMPAIGN = "rgsp_sadd_controls_20261008_v1"


def prepare(source, lane, expected_digest, smoke):
    report = audit(source)
    if report['status'] != 'PASS' or report['inventory_sha256'] != expected_digest:
        raise RuntimeError(f"Dataset audit/digest rejected: {report['issues']}; {report['inventory_sha256']}")
    core.frozen(lane / 'manifest/source_inventory.json', report['inventory'])
    core.frozen(lane / 'manifest/source_audit.json', report)
    rows = report['inventory']
    if smoke:
        rows = ([r for r in rows if r['split'] == 'train'][:32]
                + [r for r in rows if r['split'] == 'val'][:16])
    for row in rows:
        split, name = row['split'], row['name']
        for kind, filename, key in (('images', name, 'image_sha256'),
                                    ('labels', Path(name).stem + '.txt', 'label_sha256')):
            origin = source / kind / split / filename
            dest = lane / 'data' / kind / split / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(origin, dest)
            if core.sha(dest) != row[key]:
                raise RuntimeError(f"Staged data differs: {dest}")
    core.frozen(lane / 'manifest/data_content.json', rows)
    data = {'path': str(lane / 'data'), 'train': 'images/train', 'val': 'images/val',
            'names': {0: 'aircraft'}}
    if not smoke:
        data['test'] = 'images/test'
    data_path = lane / 'manifest/data.yaml'
    core.frozen(data_path, yaml.safe_dump(data, sort_keys=False))
    return data_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    p.add_argument('--expected-inventory', required=True)
    p.add_argument('--mode', choices=('prepare', 'smoke', 'run'), default='prepare')
    p.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    args = p.parse_args()
    if args.mode != 'smoke' and args.variants != list(VARIANTS):
        p.error('Formal SADD matrix must include all four prespecified variants in order')
    lane = args.output.resolve()
    lane.mkdir(parents=True, exist_ok=True)
    lock = (lane / 'lane.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (lane / 'status.json').exists():
        raise RuntimeError('Lane has already started; inspect it, do not automatically relaunch')
    smoke = args.mode == 'smoke'
    data = prepare(args.source_data.resolve(), lane, args.expected_inventory, smoke)
    configs = {}
    for variant in args.variants:
        configs[variant] = lane / 'manifest/configs' / (variant + '.yaml')
        core.frozen(configs[variant], yaml.safe_dump(core.config(variant), sort_keys=False))
    hashes = core.code_hashes()
    for name in ('audit_sadd_data.py', 'run_sadd_controls.py', 'launch_sadd_lane.sh'):
        hashes['tools/' + name] = core.sha(ROOT / 'tools' / name)
    core.frozen(lane / 'manifest/protocol.json', {
        'campaign': CAMPAIGN, 'dataset': 'SADD', 'seed': args.seed, 'smoke_only': smoke,
        'variants': args.variants, 'epochs': 2 if smoke else 200, 'patience': 40,
        'imgsz': 640, 'batch': 16, 'nbs': 64, 'optimizer': 'AdamW', 'lr0': 0.0002,
        'weights': 'none', 'amp': False, 'train_workers': 4, 'eval_workers': 0,
        'module_warmup_epochs': 0, 'cos_lr': False, 'code': hashes,
        'data_manifest_sha256': core.sha(lane / 'manifest/data_content.json'),
        'source_inventory_sha256': args.expected_inventory,
        'config_sha256': {v: core.sha(c) for v, c in configs.items()},
        'split': 'fixed recovered positive-only 795/44/44 across all seeds and variants',
        'randomness': 'training seeds 0/1/2 only; split is not resampled',
        'fitness': '0.1*AP50+0.9*AP50_95; validation-selected best.pt',
        'train_val_nms_iou': 0.7, 'test_nms_iou': 0.6, 'test_conf': 0.001,
        'initialization': 'same seed within each paired control; reset stride-probe BN statistics',
        'hyperparameters_source': 'tools/train_rgsp_control.py; all shared settings explicit',
        'comparison_boundary': 'new controlled protocol, not a reproduction of old batch32 results',
        'limitations': 'positive-only local image split; no scene-independence proof; only 44 test images',
        'failure_policy': 'stop lane on error; no automatic resume, replacement seed or hyperparameter change',
    })
    if args.mode == 'prepare':
        print(f'PREPARED {lane}', flush=True)
        return
    status = {'host': socket.gethostname(), 'pid': os.getpid(), 'seed': args.seed,
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'started_utc': datetime.now(timezone.utc).isoformat(), 'state': 'running',
              'completed': [], 'smoke_only': smoke}
    status_path = lane / 'status.json'

    def update():
        status['updated_utc'] = datetime.now(timezone.utc).isoformat()
        status_path.write_text(json.dumps(status, indent=2) + '\n')

    try:
        for variant in args.variants:
            status.update(current=variant, stage='train')
            update()
            run = lane / 'runs' / variant
            core.gpu_idle()
            command = [sys.executable, '-u', str(ROOT / 'tools/train_rgsp_control.py'),
                       '--config', str(configs[variant]), '--data', str(data),
                       '--project', str(lane / 'runs'), '--variant', variant, '--seed', str(args.seed),
                       '--epochs', '200', '--patience', '40', '--batch', '16']
            if smoke:
                command.append('--smoke')
            core.execute(command, lane / 'logs' / (variant + '.train.log'))
            completion = json.loads((run / 'train_complete.json').read_text())
            if completion['best_sha256'] != core.sha(run / 'weights/best.pt'):
                raise RuntimeError('Checkpoint hash differs from training completion evidence')
            status['stage'] = 'evaluate_val_smoke' if smoke else 'evaluate_test'
            update()
            core.gpu_idle()
            eval_path = run / ('eval_smoke_val.json' if smoke else 'eval_test.json')
            core.execute([
                sys.executable, '-u', str(ROOT / 'tools/evaluate_sarmix.py'),
                '--weights', str(run / 'weights/best.pt'), '--data', str(data),
                '--experiment', f'{CAMPAIGN}_seed{args.seed}_{variant}',
                '--placement', 'baseline' if variant == 'original' else 'p3',
                '--variant', variant, '--state-dim', '0' if variant == 'original' else '16',
                '--seed', str(args.seed), '--subset', 'SADD', '--split', 'val' if smoke else 'test',
                '--output-json', str(eval_path), '--imgsz', '640', '--batch', '16',
                '--device', '0', '--workers', '0', '--conf', '0.001', '--iou', '0.6',
                '--project', str(lane / 'evaluations'), '--name', variant,
            ], lane / 'logs' / (variant + '.eval.log'))
            result = json.loads(eval_path.read_text())
            if result['seed'] != args.seed or result['weights_sha256'] != completion['best_sha256']:
                raise RuntimeError('Evaluation identity mismatch')
            core.frozen(run / 'evaluation_complete.json', {
                'exit_code': 0, 'eval_sha256': core.sha(eval_path), 'smoke_only': smoke,
                'finished_utc': datetime.now(timezone.utc).isoformat()})
            status['completed'].append(variant)
            if not smoke:
                core.summarize(lane)
        status.update(state='complete', stage='finished')
    except Exception as error:
        status.update(state='failed', error=repr(error))
        raise
    finally:
        update()


if __name__ == '__main__':
    main()
