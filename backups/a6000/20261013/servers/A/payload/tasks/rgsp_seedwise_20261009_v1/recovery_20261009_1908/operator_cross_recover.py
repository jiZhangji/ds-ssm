#!/usr/bin/env python3
"""One finite operator recovery of A's fifteen already-planned seed0 cross runs.

Not part of the immutable campaign source list. No retry, data staging, AIR run,
seed1 dispatch, barrier publication, process signal, installation or service.
All formal arguments are forwarded unchanged through the frozen UUID launcher.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

TASK = Path('/home/powerleader/ig-ds-ssm/tasks/rgsp_seedwise_20261009_v1')
OUT = TASK / 'recovery_20261009_1908'
EXPECTED_PLAN = '9c6b1afe025155c2209e56ecd745a7fccccd82dc96b4bd75987c66a709bfd7e4'
EXPECTED_PLAN_FILE = '37e1e2254643541118e4bab29d11cb93f1483847fc73aa69be8d63be4907cbe0'
EXPECTED_WRAPPER = '3faa16696535c447d50a405799040e17ef7780bb9444c3f5b0c9d5310061b04d'
EXPECTED_CROSS = 'e348f28b38c41a271fed5776510e2b44b241c20f8399eb86ee8aca5fad7d5e5f'
GPU = 'GPU-12cf4841-46b9-94d1-2715-9932134cf964'
HOST = 'powerleader-PT6620W'
PREFIXES = ('sar_acd', 'sample', 'ottawa', 'farmland', 'yellow_river')
VARIANTS = ('original', 'matched_scale075', 'full')
SELECTED = tuple(f'{dataset}_{variant}_seed0' for dataset in PREFIXES for variant in VARIANTS)
PREDECESSORS = ('sadd_matched_scale075_seed0', 'sadd_full_seed0', 'sadd_local_energy_seed0',
                'sadd_matched_identity_seed0', 'sadd_cnn_yolov8n_seed0')
PENDING_AIR = ('sar_aircraft10_original_seed0', 'sar_aircraft10_full_seed0')
SCHEMA = 'rgsp-operator-cross-recovery-v1'

sys.path.insert(0, str(TASK))
import manifest as m
from campaign_dispatch import process_identity, identity_alive
from run_lane import lock_file, wait_idle, gpu_snapshot
from verify_receipts import expected_artifacts


def arg(command, flag):
    if command.count(flag) != 1 or command.index(flag) + 1 >= len(command):
        raise RuntimeError(f'Expected one valued flag: {flag}')
    return command[command.index(flag) + 1]


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError('Nonfinite number in completion evidence')
    if isinstance(value, dict):
        for child in value.values():
            finite(child)
    if isinstance(value, (list, tuple)):
        for child in value:
            finite(child)


def snapshot_file(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f'Expected regular, non-symlink evidence file: {path}')
    return {'path': str(path), 'size': path.stat().st_size, 'sha256': m.sha(path)}


def no_training_output(trial):
    """This recovery does not silently resume partially trained experiments."""
    root = Path(trial['run_dir'])
    if root.is_symlink():
        raise RuntimeError(f'Symlink run directory refused: {root}')
    existing = []
    if root.exists():
        for path in sorted(root.rglob('*')):
            if path.is_symlink() or path.is_dir() or path.name not in ('config.json', 'initialization.json') or path.parent != root:
                raise RuntimeError(f'Existing training/unknown output requires separate resume audit: {path}')
            existing.append(snapshot_file(path))
    return existing


def verify_receipt(plan, trial):
    path = m.receipt_path(plan, trial)
    receipt = m.validate_receipt(plan, trial, m.read_json(path), verify_files=True)
    if [item['path'] for item in receipt['artifacts']] != expected_artifacts(trial):
        raise RuntimeError('Existing receipt has a different ordered artifact contract')
    return snapshot_file(path)


def verify_inputs():
    if Path(__file__).resolve() != OUT / 'operator_cross_recover.py' or socket.gethostname() != HOST:
        raise RuntimeError('This operator is pinned to the approved A task directory and host')
    plan = m.load_plan(TASK / 'plan.json')
    if m.digest(plan) != EXPECTED_PLAN or m.sha(TASK / 'plan.json') != EXPECTED_PLAN_FILE or len(plan['code_hashes']) != 360:
        raise RuntimeError('Frozen plan or 360-entry source inventory changed')
    m.verify_code(plan)
    if m.sha(TASK / 'cuda_uuid_compat.py') != EXPECTED_WRAPPER or m.sha(TASK / 'cross_trial.py') != EXPECTED_CROSS:
        raise RuntimeError('Frozen compatibility helper/trainer changed')
    lane = next(row for row in plan['lanes'] if row['id'] == 'a_seed0')
    if lane['hostname'] != HOST or lane['gpu_uuid'] != GPU or tuple(lane['trial_ids']) != PREDECESSORS + SELECTED + PENDING_AIR:
        raise RuntimeError('A seed0 lane differs from this reviewed recovery scope')
    if plan['dispatch_map']['a']['seed0_lane'] != lane['id']:
        raise RuntimeError('Dispatch lane changed')
    old_dispatch = m.read_json(TASK / 'dispatch/a/status.json')
    old_lane = m.read_json(Path(lane['output_dir']) / 'status.json')
    if old_dispatch['state'] != 'failed' or old_lane['state'] != 'failed':
        raise RuntimeError('Recovery requires the preserved failed original dispatch/lane')
    for key in ('owner', 'child'):
        if identity_alive(old_dispatch.get(key)):
            raise RuntimeError('Original dispatcher or child is still alive')
    predecessors = [verify_receipt(plan, m.trial_by_id(plan, tid)) for tid in PREDECESSORS]
    selected = []
    for tid in SELECTED:
        trial = m.trial_by_id(plan, tid)
        command = trial['command']
        if trial['kind'] != 'command' or trial['seed'] != 0 or trial.get('assigned_host_key') != 'a':
            raise RuntimeError('A recovery is strictly seed0 command trials only')
        if command[:2] != [plan['python'], str(TASK / 'cross_trial.py')]:
            raise RuntimeError('Not the frozen cross-trial command')
        if arg(command, '--seed') != '0' or '--validate-only' in command or command.count('--resume') != 1:
            raise RuntimeError('Formal seed/resume/validate-only flags changed')
        if arg(command, '--plan') != str(TASK / 'plan.json') or arg(command, '--trial-id') != tid:
            raise RuntimeError('Formal plan/trial binding changed')
        if arg(command, '--output') != trial['run_dir'] or arg(command, '--source-root') != plan['source_root']:
            raise RuntimeError('Formal source/output binding changed')
        m.within(trial['run_dir'], TASK / 'runs')
        if m.receipt_path(plan, trial).exists():
            raise RuntimeError(f'Previously completed cross trial requires new reviewed recovery scope: {tid}')
        selected.append({'trial_id': tid, 'trial_sha256': m.digest(trial), 'planned_argv': command,
                         'existing_metadata': no_training_output(trial)})
    failure_paths = [TASK / 'dispatch/a/status.json', Path(lane['output_dir']) / 'status.json',
                     Path(lane['output_dir']) / 'logs/sar_acd_original_seed0.log']
    return plan, lane, {'schema': SCHEMA, 'checked_utc': m.utc(), 'hostname': HOST,
                        'plan_sha256': EXPECTED_PLAN, 'plan_file_sha256': EXPECTED_PLAN_FILE,
                        'source_files_verified': 360, 'predecessor_receipts': predecessors,
                        'selected': selected, 'failure_evidence': [snapshot_file(p) for p in failure_paths],
                        'pending_air_ids': list(PENDING_AIR), 'no_seed1_authorization': True}


def runtime_env(plan):
    env = os.environ.copy()
    updates = {'CUDA_VISIBLE_DEVICES': GPU, 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1',
               'PYTHONPATH': plan['source_root'], 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
               'OPENBLAS_NUM_THREADS': '4', 'YOLO_AUTOINSTALL': 'false', 'WANDB_MODE': 'disabled',
               'YOLO_CONFIG_DIR': str(OUT / 'runtime-config'), 'MPLCONFIGDIR': str(OUT / 'matplotlib')}
    env.update(updates)
    return env, updates


def wrapper_command(planned, compat_output):
    actual = [planned[0], '-B', '-u', str(TASK / 'cuda_uuid_compat.py'), '--expected-uuid', GPU,
              '--script', planned[1], '--output', str(compat_output), '--', *planned[2:]]
    if actual[actual.index('--') + 1:] != planned[2:]:
        raise RuntimeError('Not an exact argument-preserving wrapper')
    return actual


def verify_compat(path):
    report = m.read_json(path)
    identity = report.get('cuda_identity', {})
    if report.get('status') != 'PASS' or report.get('script_sha256') != EXPECTED_CROSS or report.get('helper_sha256') != EXPECTED_WRAPPER:
        raise RuntimeError('Compatibility launcher did not successfully run the frozen cross script')
    if identity.get('hostname') != HOST or identity.get('physical_gpu_uuid') != GPU or identity.get('cuda_visible_devices') != GPU:
        raise RuntimeError('Compatibility report GPU/host identity mismatch')
    if identity.get('reported_count_after') != 1 or identity.get('cuda_runtime_count') != 1 or identity.get('gpu_mask_changed') is not False or identity.get('torch_installation_modified') is not False:
        raise RuntimeError('Compatibility report changed the GPU mask or installation')
    return snapshot_file(path)


def launch(plan, lane, planned, label, locks, status, mode):
    evidence = OUT / mode / label
    evidence.mkdir(parents=True, exist_ok=False)
    compat = evidence / 'cuda_compatibility.json'
    actual = wrapper_command(planned, compat)
    env, updates = runtime_env(plan)
    m.verify_code(plan)
    status.update(current=label, phase=mode)
    idle = wait_idle(GPU, lane, OUT / f'{mode}_status.json', status)
    if shutil.disk_usage(TASK).free < 20 * 1024 ** 3:
        raise RuntimeError('Less than 20GiB available; no training started')
    # Check once again after all pre-launch I/O, without whitelisting compute PIDs.
    fresh = gpu_snapshot(GPU)
    if fresh['compute_processes'] or fresh['memory_used_mib'] > lane['max_idle_memory_mib'] or fresh['utilization_percent'] > lane['max_idle_utilization_percent']:
        raise RuntimeError('GPU became occupied after idle gate; no process interrupted')
    contract = {'schema': SCHEMA, 'mode': mode, 'label': label, 'planned_argv': planned, 'actual_argv': actual,
                'forwarded_args_exact': actual[actual.index('--') + 1:] == planned[2:],
                'plan_sha256': EXPECTED_PLAN, 'plan_file_sha256': EXPECTED_PLAN_FILE,
                'wrapper_sha256': EXPECTED_WRAPPER, 'cross_trial_sha256': EXPECTED_CROSS,
                'operator_sha256': m.sha(__file__), 'environment_overrides': updates,
                'cwd': plan['source_root'], 'owner': process_identity(os.getpid()),
                'lock_paths': [str(handle.name) for handle in locks], 'child_inherits_all_locks': True,
                'idle_gate': idle, 'fresh_gpu': fresh, 'started_utc': m.utc()}
    m.frozen(evidence / 'launch_contract.json', contract)
    with (evidence / 'stdout.log').open('x', encoding='utf-8') as stream:
        child = subprocess.Popen(actual, cwd=plan['source_root'], env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, pass_fds=tuple(handle.fileno() for handle in locks))
        status.update(state='running', child=process_identity(child.pid), launch_contract=str(evidence / 'launch_contract.json'), updated_utc=m.utc())
        m.atomic_json(OUT / f'{mode}_status.json', status)
        returncode = child.wait()
    m.frozen(evidence / 'exit.json', {'returncode': returncode, 'finished_utc': m.utc()})
    if returncode != 0:
        raise RuntimeError(f'{label} exited {returncode}; preserve output, stop queue without retry')
    verify_compat(compat)
    return evidence


def verify_cross_completion(plan, trial):
    """Recursively verify the internal completion ledger, not just two outer files."""
    root = Path(trial['run_dir'])
    cfg = m.read_json(root / 'config.json')
    completion = m.read_json(root / 'completion.json')
    result = m.read_json(root / 'result.json')
    tested = m.read_json(root / 'test_evaluation.json')
    identity = m.digest(cfg)
    variant = arg(trial['command'], '--variant')
    dataset = arg(trial['command'], '--dataset')
    if cfg['seed'] != 0 or cfg['variant'] != variant or cfg['trial_id'] != trial['id'] or cfg['validate_only'] is not False or cfg['plan_sha256'] != m.digest(plan):
        raise RuntimeError('Completed config is not this formal seed0 trial')
    if cfg['source_root'] != plan['source_root'] or cfg['protocol']['dataset'] != dataset:
        raise RuntimeError('Completed source/dataset differs from frozen trial')
    for flag in ('epochs', 'patience', 'batch', 'imgsz'):
        if str(cfg['protocol'][flag]) != arg(trial['command'], '--' + flag):
            raise RuntimeError(f'Completed protocol differs: {flag}')
    protocol_hash = m.digest(cfg['protocol'])
    if cfg['protocol_sha256'] != protocol_hash or completion['protocol_sha256'] != protocol_hash or result['protocol_sha256'] != protocol_hash or result['protocol'] != cfg['protocol']:
        raise RuntimeError('Protocol hash or contents mismatch')
    if completion['status'] != 'complete' or completion['identity_sha256'] != identity or completion['seed'] != 0 or completion['variant'] != variant:
        raise RuntimeError('Internal completion identity mismatch')
    names = ('config.json', 'initialization.json', 'metrics.jsonl', 'test_evaluation.json', 'result.json', 'best.pt')
    wanted = [str(root / name) for name in names]
    if [item['path'] for item in completion['artifacts']] != wanted:
        raise RuntimeError('Internal completion must cover exactly the ordered six semantic artifacts')
    for item in completion['artifacts']:
        snapshot_file(item['path'])
        m.check_file(item)
    best_sha = m.sha(root / 'best.pt')
    if result['best_sha256'] != best_sha or tested['best_sha256'] != best_sha or tested['identity_sha256'] != identity:
        raise RuntimeError('Best/test/result checkpoint binding mismatch')
    if result['seed'] != 0 or result['variant'] != variant or result['dataset'] != dataset or result['checkpoint'] != str(root / 'best.pt'):
        raise RuntimeError('Result describes a different trial')
    if result['metric'] != tested['metric'] or result['threshold'] != tested['threshold'] or result['best_epoch'] != tested['checkpoint_epoch']:
        raise RuntimeError('Test/result evidence mismatch')
    rows = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    if not rows or [row['epoch'] for row in rows] != list(range(1, len(rows) + 1)) or not (1 <= result['best_epoch'] <= len(rows)):
        raise RuntimeError('Epoch ledger is incomplete, duplicated or inconsistent with selected best')
    if len(rows) > cfg['protocol']['epochs']:
        raise RuntimeError('Epoch ledger exceeded frozen budget')
    for row in rows:
        if not isinstance(row['loss'], (int, float)):
            raise RuntimeError('Missing numeric training loss')
    finite([cfg, completion, result, tested, rows])
    return {'schema': SCHEMA, 'status': 'deep_verified', 'trial_id': trial['id'], 'seed': 0,
            'plan_sha256': m.digest(plan), 'identity_sha256': identity, 'protocol_sha256': protocol_hash,
            'internal_artifacts': completion['artifacts'], 'last_checkpoint': snapshot_file(root / 'last.pt'),
            'actual_epochs': len(rows), 'best_epoch': result['best_epoch'], 'best_sha256': best_sha,
            'finite_training_and_test': True, 'checked_utc': m.utc()}


def probes(plan, lane, locks, status):
    for tid, label in (('sar_acd_original_seed0', 'classification'), ('ottawa_original_seed0', 'change_detection')):
        command = list(m.trial_by_id(plan, tid)['command'])
        for flag in ('--plan', '--trial-id'):
            index = command.index(flag)
            del command[index:index + 2]
        command.remove('--resume')
        command[command.index('--output') + 1] = str(OUT / 'probe_runs' / label)
        command[command.index('--workers') + 1] = '0'
        command.append('--validate-only')
        evidence = launch(plan, lane, command, label, locks, status, 'probes')
        validation = m.read_json(OUT / 'probe_runs' / label / 'validation.json')
        if validation.get('status') != 'passed' or validation.get('validation_only') is not True or len(validation['checks']) != 3:
            raise RuntimeError('Independent three-arm forward/backward/reload probe failed')
        finite(validation)
        m.frozen(evidence / 'verified.json', {'validation': snapshot_file(OUT / 'probe_runs' / label / 'validation.json'),
                                          'compatibility': verify_compat(evidence / 'cuda_compatibility.json'),
                                          'formal_receipt_written': False})
        status['completed'].append(label)


def run_queue(plan, lane, locks, status):
    for label in ('classification', 'change_detection'):
        record = m.read_json(OUT / 'probes' / label / 'verified.json')
        m.check_file(record['validation'])
        m.check_file(record['compatibility'])
        verify_compat(record['compatibility']['path'])
    for tid in SELECTED:
        trial = m.trial_by_id(plan, tid)
        no_training_output(trial)
        if m.receipt_path(plan, trial).exists():
            raise RuntimeError('Unexpected new receipt; no duplicate execution')
        evidence = launch(plan, lane, trial['command'], tid, locks, status, 'trials')
        audit = verify_cross_completion(plan, trial)
        m.frozen(evidence / 'deep_completion_audit.json', audit)
        contract = m.read_json(evidence / 'launch_contract.json')
        details = {'kind': 'command', 'operator_recovery': True, 'recovery_schema': SCHEMA,
                   'planned_argv': trial['command'], 'actual_argv': contract['actual_argv'],
                   'forwarded_args_exact': True, 'operator_sha256': contract['operator_sha256'],
                   'wrapper_sha256': EXPECTED_WRAPPER, 'cross_trial_sha256': EXPECTED_CROSS,
                   'launch_contract': snapshot_file(evidence / 'launch_contract.json'),
                   'compatibility_report': snapshot_file(evidence / 'cuda_compatibility.json'),
                   'deep_completion_audit': snapshot_file(evidence / 'deep_completion_audit.json'),
                   'limitations': 'Same frozen arguments through audited process-local torch1.13 UUID adapter; original argv was not directly executed. Not a bitwise-equivalence claim.'}
        m.write_receipt(plan, trial, trial['success_artifacts'], details)
        verify_receipt(plan, trial)
        status['completed'].append(tid)
        status.update(state='trial_complete', updated_utc=m.utc())
        m.atomic_json(OUT / 'trials_status.json', status)
    m.frozen(OUT / 'operator_cross_complete.json', {'schema': SCHEMA, 'plan_sha256': EXPECTED_PLAN,
               'completed_cross_ids': list(SELECTED), 'pending_air_ids': list(PENDING_AIR),
               'entire_lane_complete': False, 'global_seed0_complete': False,
               'seed1_started': False, 'finished_utc': m.utc()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('verify-only', 'probe', 'run'))
    args = parser.parse_args()
    plan, lane, baseline = verify_inputs()
    if args.mode == 'verify-only':
        print(json.dumps(baseline, indent=2))
        return
    mode = 'probes' if args.mode == 'probe' else 'trials'
    marker = OUT / f'{mode}_launch.json'
    if marker.exists() or (OUT / f'{mode}_status.json').exists():
        raise RuntimeError('This finite attempt already started; inspect before any new recovery')
    locks = []
    status = {'schema': SCHEMA, 'owner': process_identity(os.getpid()), 'mode': mode,
              'plan_sha256': EXPECTED_PLAN, 'completed': [], 'started_utc': m.utc(), 'state': 'starting'}
    try:
        for path in (TASK / 'dispatch/a/dispatch.lock', Path(lane['output_dir']) / 'lane.lock', TASK / 'locks' / (GPU + '.lock')):
            locks.append(lock_file(path))
        # Repeat immutable/failed-state/output checks after acquiring all locks.
        plan, lane, baseline = verify_inputs()
        m.frozen(marker, {**baseline, 'operator_sha256': m.sha(__file__), 'owner': process_identity(os.getpid())})
        if args.mode == 'probe':
            probes(plan, lane, locks, status)
        else:
            run_queue(plan, lane, locks, status)
        status.update(state='complete', finished_utc=m.utc())
    except BaseException as error:
        status.update(state='failed', error=f'{type(error).__name__}: {error}', failed_utc=m.utc())
        raise
    finally:
        status['updated_utc'] = m.utc()
        m.atomic_json(OUT / f'{mode}_status.json', status)
        for handle in reversed(locks):
            handle.close()


if __name__ == '__main__':
    main()
