#!/usr/bin/env python3
"""Create reviewed immutable smoke/formal plans; never starts a model or job."""
import argparse
import json
from pathlib import Path

from manifest import SCHEMA, config_payload, digest, frozen, load_plan, sha

ROOT = Path('/home/powerleader/ig-ds-ssm/tasks/rgsp_seedwise_20261009_v1')
PYTHON = '/home/powerleader/ig-ds-ssm/envs/ig-ds-ssm-env/bin/python'
HOSTS = {
    'a': ('powerleader-PT6620W', 'GPU-12cf4841-46b9-94d1-2715-9932134cf964'),
    'b': ('powerleader-a6000-b', 'GPU-09b018e1-7140-296a-73d4-9e010ee209a2'),
    'c': ('a6000-c', 'GPU-e5f8a5d2-306d-582e-c926-f795df943b1b'),
    'd': ('a6000-d', 'GPU-f1b51ae3-c020-9901-c608-cca67f6ab8f1'),
    'f0': ('powerleader-a6000-f', 'GPU-9d27eec3-fd30-8534-8341-b983bea3832b'),
    'f1': ('powerleader-a6000-f', 'GPU-ce57e76d-ee2e-ff91-dbdd-ec0938874c40'),
}


def source_hashes():
    files = sorted(p for p in (ROOT / 'code').rglob('*')
                   if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    files += [ROOT / name for name in ('manifest.py', 'run_lane.py', 'train_trial.py',
              'cross_data.py', 'cross_trial.py', 'cuda_uuid_compat.py')]
    for name in ('adopt_receipt.py', 'campaign_dispatch.py', 'deferred_data.py', 'verify_receipts.py'):
        if (ROOT / name).is_file():
            files.append(ROOT / name)
    return [{'path': str(p), 'sha256': sha(p)} for p in files]


def datasets():
    return {
        'ssdd': {'protocol': 'ssdd', 'root': str(ROOT / 'data/ssdd/data'),
                 'yaml': str(ROOT / 'data/ssdd/data.yaml'),
                 'yaml_sha256': '55784492a13eec556af419b736b7005a3f308dba96b1f4df2b02883199af34ec',
                 'manifest': str(ROOT / 'data/ssdd/data_content.json'),
                 'manifest_sha256': 'cd2f5314ecc8139f8e51a8c283d309f522c12abd58320003ea86bacbaac853de',
                 'counts': {'train': 730, 'val': 186, 'test': 232}},
        'sadd': {'protocol': 'sadd', 'root': str(ROOT / 'data/sadd/data'),
                 'yaml': str(ROOT / 'data/sadd/data.yaml'),
                 'yaml_sha256': 'de31fa526be538df8b36d1019e4a72edb99198fb1b67c69f4a61faed4f1d4122',
                 'manifest': str(ROOT / 'data/sadd/data_content.json'),
                 'manifest_sha256': '23077fb38483e161a3b7675b1d0629b19cbe334482588a0bdebf7eba88372e10',
                 'counts': {'train': 795, 'val': 44, 'test': 44}},
    }


def detection(root, ds, control, seed=0, label=None, **options):
    tid = f'{ds}_{label or control}_seed{seed}'
    spec = {'control': control, 'placement': 'baseline' if control in ('original', 'cnn_yolov8n') else 'p3',
            'state_dim': 0 if control in ('original', 'cnn_yolov8n') else 16,
            'trend_kernel_size': 3, 'stats_kernel_size': 3, **options}
    config_payload(spec)
    return {'id': tid, 'seed': seed, 'kind': 'detection', 'dataset': ds, 'spec': spec,
            'config': str(root / 'configs' / (tid + '.yaml')), 'run_dir': str(root / 'runs' / tid)}


def lane(root, name, tids):
    hostname, uuid = HOSTS[name]
    return {'id': name, 'hostname': hostname, 'gpu_uuid': uuid,
            'output_dir': str(root / 'lanes' / name), 'idle_seconds': 20,
            'idle_timeout_seconds': 172800, 'max_idle_memory_mib': 2500,
            'max_idle_utilization_percent': 10, 'trial_ids': tids}


def base(root):
    return {'schema': SCHEMA, 'campaign': root.name, 'campaign_root': str(root),
            'source_root': str(ROOT / 'code'), 'python': PYTHON,
            'barrier_path': str(root / 'seed0_barrier.json'), 'receipts_dir': str(root / 'receipts'),
            'code_hashes': source_hashes(), 'datasets': datasets(), 'trials': [], 'lanes': [],
            'seed_policy': 'Entire intended seed0 matrix before any new seed1; seed2 forbidden.',
            'split_policy': 'Both new training seeds use frozen SSDD seed0 split and fixed SADD/cross-task splits.',
            'initialization_boundary': 'Native seeded scratch for detection; common tensors paired within equal-shape controls and windows, not across architecture/state/placement.',
            'legacy_boundary': 'Already-running F seed1 LocalEnergy may finish, is preserved separately and cannot launch subsequent seed1.'}


def smoke(host):
    root = ROOT / ('smoke_' + host + '_v1')
    plan = base(root)
    plan['smoke_only'] = True
    if host == 'b':
        trials = [detection(root, 'ssdd', c) for c in ('full_no_lrc', 'const_guidance', 'post_gate')]
    elif host == 'c':
        trials = [detection(root, 'ssdd', 'full', label='p4', placement='p4'),
                  detection(root, 'ssdd', 'full', label='d8', state_dim=8),
                  detection(root, 'ssdd', 'cnn_yolov8n')]
    elif host == 'a':
        trials = [detection(root, 'sadd', 'original')]
    else:
        trials = [detection(root, 'sadd', 'full_no_lrc')]
    plan['trials'] = trials
    plan['seed0_trial_ids'] = [t['id'] for t in trials]
    plan['lanes'] = [lane(root, host, plan['seed0_trial_ids'])]
    return root / 'plan.json', plan


CROSS_HASHES = {
    'SAR-ACD': ('SAR_ACD', '16259205a8c8e0677631e37e23bba036311ea87cef8cf6f833b0f626a303c259', '567b09e122db10024e281b047e79417ba142a8ddf90bae2754bc640959dd6db4', '5fcd4daa8afdfedd6f519de4d4c2e7ec797f90f6904fa2a61af964f111205b93'),
    'SAMPLE': ('SAMPLE', '3b34a6d7c88e6d80c71b429082fb8af6652c5231f716e63bd5c8bc1bdb88c939', '15350dd6b82931f37ac4f9409aa770f0d75a5541276932c654d01433b685dc3c', 'a50fba37e9f4a148b5fd66884f261670e0526d882dde4febfb820da967ced87a'),
    'Ottawa': ('Ottawa', '7cfc014e6c6e94a670b692382c1fe5328404035e3ceb29941f62aef4209721d7', 'f487199e1d2f4ee9fc7150a1614a670ecbf2fd1e31aa993f1e688dfba28086fc', '5935f8221f4c420c908ee2b67be062ac706bfc881051f5ecdfe0081676d4a8cd'),
    'Farmland': ('Farmland', '5c5e7521398ab6575e98eb986afdfb4dbb8c595726351b0d111d1ee333f7ba2b', 'cfb05ae7d9b736534cc4a5bd95831fa14f73025e63dd4aeea704913e7f3af2d6', '6e30aa25bbf59ce5799dacdc93bf836f60bc03fb1438ed90b4681b7bafb2ec7b'),
    'Yellow_River': ('Yellow_River', 'a7aa8b3d0e78ff7a8364545409cf73732e264d2946ff2e9fcfee98bdce77d224', '44ddd56e277fa71e0795de6a6cf9ed9c220798aad27ab770e967211c241eec2a', 'e9d4f235c47f3f2c3add54ff364dded861f780eb2b1f76df4e02935d8bc37679'),
}


def cross_trial(plan, dataset, variant, seed):
    folder = CROSS_HASHES[dataset][0]
    key = {'SAR-ACD': 'sar_acd', 'Yellow_River': 'yellow_river'}.get(dataset, dataset.lower())
    tid = f'{key}_{variant}_seed{seed}'
    output = ROOT / 'runs' / tid
    classification = dataset in ('SAR-ACD', 'SAMPLE')
    command = [PYTHON, str(ROOT / 'cross_trial.py'), '--source-root', str(ROOT / 'code'),
               '--manifest', str(ROOT / 'cross_data' / folder / 'manifest.json'),
               '--output', str(output), '--dataset', dataset, '--variant', variant, '--seed', str(seed),
               '--epochs', '200' if classification else '150', '--patience', '40' if classification else '25',
               '--batch', '64' if classification else '16', '--imgsz', '224' if classification else '192',
               '--device', 'cuda:0', '--workers', '4', '--plan', str(ROOT / 'plan.json'), '--trial-id', tid, '--resume']
    return {'id': tid, 'seed': seed, 'kind': 'command', 'dataset': dataset,
            'command': command, 'cwd': str(ROOT / 'code'), 'run_dir': str(output),
            'success_artifacts': [str(output / n) for n in ('result.json', 'completion.json')]}


def formal():
    plan = base(ROOT)
    reuse = json.loads((ROOT / 'reuse_entries.json').read_text())
    plan['cross_datasets'] = {name: {'manifest': str(ROOT / 'cross_data' / item[0] / 'manifest.json'),
        'manifest_sha256': item[1], 'files_sha256': item[2], 'audit_sha256': item[3]}
        for name, item in CROSS_HASHES.items()}
    plan['dispatch_map'] = {}
    plan['resource_policy'] = {'active_keys': ['a', 'b', 'c', 'f0', 'f1'],
        'd': 'Excluded from new dispatch: unrelated scaf-grpo compute appeared after preflight; do not interrupt.',
        'other_registered_hosts': 'Busy, no adequate disk, or unreachable at inventory; no takeover or local fallback.'}
    plan['scheduling_policy'] = 'Reserve estimated long external workloads first, then balance eligible core work around them; estimates are scheduling weights, not completion-time promises.'
    plan['external_protocol'] = 'New same-budget scratch 150e/patience30/640/BS16/FP32/AdamW controls; never mixed with old unmatched 300e/patch claims.'
    for ds, identity, nc in (('sar_shipdataset', 'SAR-ShipDataset', 1), ('sar_aircraft10', 'SAR-AIRcraft1.0', 7)):
        dr = ROOT / 'data' / ds
        plan['datasets'][ds] = {'deferred': True, 'descriptor': str(dr / 'dataset.json'),
            'audit': str(dr / 'source_audit.json'), 'readiness': str(dr / 'readiness.json'),
            'expected_identity': identity, 'expected_nc': nc, 'protocol': 'ssdd'}
    for seed in (0, 1):
        work = []
        def add(ds, control, label=None, **options):
            t = detection(ROOT, ds, control, seed, label, **options)
            work.append(t)
            return t
        # Primary module evidence first; shared existing Full is reused on SADD.
        for ds, c in [('ssdd','full'), ('sadd','full_no_lrc'), ('ssdd','full_no_lrc'),
                      ('sadd','const_guidance'), ('ssdd','const_guidance'), ('sadd','post_gate'),
                      ('ssdd','post_gate'), ('sadd','original'), ('sadd','matched_identity'),
                      ('ssdd','matched_scale075')]:
            add(ds,c)
        for c in ('original', 'matched_identity', 'local_energy', 'local_energy_input',
                  'local_energy_delta', 'full_fixed_half', 'residual_scale075', 'energy_only',
                  'guidance_only', 'global', 'lsgm_trend_only', 'lsgm_std_only', 'lsgm_point_only'):
            add('ssdd',c)
        for c in ('matched_scale075','full','local_energy'):
            add('sadd',c)
        for d in (8,32):
            add('ssdd','full',label=f'full_d{d}',state_dim=d)
        for k in (5,7):
            add('ssdd','full',label=f'full_trend{k}',trend_kernel_size=k)
            add('ssdd','full',label=f'full_stats{k}',stats_kernel_size=k)
        for loc in ('p2','p4'):
            add('ssdd','full',label=f'full_{loc}',placement=loc)
        for ds in ('ssdd','sadd'):
            add(ds,'cnn_yolov8n')
        for dataset in CROSS_HASHES:
            for variant in ('original','matched_scale075','full'):
                work.append(cross_trial(plan,dataset,variant,seed))
        # Delayed-data commands stay in the global completion set; no seed gate shortcut.
        for ds in ('sar_shipdataset','sar_aircraft10'):
            for variant in ('original','matched_scale075','full'):
                t = add(ds,variant)
                t.update(kind='command',runner_kind='deferred_detection',cwd=str(ROOT / 'code'))
                t['success_artifacts'] = [str(Path(t['run_dir']) / n) for n in (
                    'provenance.json','train_complete.json','eval_test.json','evaluation_complete.json',
                    'weights/best.pt','weights/raw_fp32_last.pt','controls_by_epoch.jsonl','trial_complete.json')]
        external_assignment = {
            f'sar_shipdataset_original_seed{seed}': 'f0',
            f'sar_shipdataset_matched_scale075_seed{seed}': 'f1',
            f'sar_shipdataset_full_seed{seed}': 'c',
            f'sar_aircraft10_original_seed{seed}': 'a',
            f'sar_aircraft10_matched_scale075_seed{seed}': 'b',
            f'sar_aircraft10_full_seed{seed}': 'a',
        }
        # Long jobs dominate the makespan. Charging them only after all short
        # assignments leaves A/B underused while C/F carry both heavy and short
        # workloads. Reserve their reviewed host/data placement before filling
        # core work; actual lane order still puts immediately ready work first.
        loads = {'a':1200,'b':600,'c':2400,
                 'f0':2400+(30 if seed==0 else 0),'f1':2400+(15 if seed==0 else 0)}
        assignment = {h:[] for h in loads}
        for t in work:
            tid, ds = t['id'],t['dataset']
            if tid in reuse:
                t.update(kind='reuse',reuse=reuse[tid])
                host = ('a' if seed == 0 else 'b') if ds == 'sadd' else 'f0'
                cost = 0
            elif tid == 'ssdd_local_energy_seed0':
                host,cost = 'f0',0
                out = Path(t['run_dir'])
                t.update(kind='command',cwd=str(ROOT / 'code'),
                    command=[PYTHON,str(ROOT / 'adopt_receipt.py'),'--plan',str(ROOT / 'plan.json'),'--trial',tid,
                             '--legacy-completion',str(ROOT / 'evidence/legacy_handoff_f_r2/f0.complete.json')],
                    success_artifacts=[str(out / n) for n in ('adoption.json','train_complete.json','eval_test.json','weights/best.pt','completion.json')])
            elif tid in external_assignment:
                host, cost = external_assignment[tid], 0
                t['command'] = [PYTHON,str(ROOT / 'train_trial.py'),'--plan',str(ROOT / 'plan.json'),
                                '--trial',tid,'--gpu-uuid',HOSTS[host][1]]
            else:
                if ds == 'sadd':
                    eligible,cost = ('a','b'),90
                elif ds == 'ssdd':
                    eligible,cost = ('b','c','f0','f1'),70
                elif ds in CROSS_HASHES:
                    eligible,cost = ('a','b','c'),20
                else:
                    raise RuntimeError(f'Unreserved external workload: {tid}')
                if t.get('spec',{}).get('control') == 'cnn_yolov8n':
                    cost = 25
                host = min(eligible,key=lambda h:(loads[h],h))
            t['assigned_host_key'] = host
            loads[host] += cost
            assignment[host].append(tid)
        plan['trials'] += work
        for host,tids in assignment.items():
            # Reuse receipts can be verified without waiting behind fresh GPU work.
            lookup = {t['id']:t for t in work}
            tids.sort(key=lambda tid: 0 if lookup[tid]['kind']=='reuse' else 1)
            item = lane(ROOT,host,tids)
            item['id'] = f'{host}_seed{seed}'
            item['output_dir'] = str(ROOT / 'lanes' / item['id'])
            plan['lanes'].append(item)
    plan['seed0_trial_ids'] = [t['id'] for t in plan['trials'] if t['seed']==0]
    for host in ('a','b','c','f0','f1'):
        item = {'hostname':HOSTS[host][0],'gpu_uuid':HOSTS[host][1],
                'seed0_lane':f'{host}_seed0','seed1_lane':f'{host}_seed1','requires':[]}
        if host in ('f0','f1'):
            item['requires'] = [{'path':str(ROOT / f'evidence/legacy_handoff_f_r2/{host}.complete.json'),
                                 'equals':{'status':'adopted_and_scheduler_retired'}}]
            item['finalizer_launch'] = str(ROOT / f'evidence/legacy_handoff_f_r2/{host}.r3.finalizer_launch.json')
        plan['dispatch_map'][host] = item
    if len(plan['seed0_trial_ids']) != 57 or len(plan['trials']) != 114:
        raise RuntimeError('Reviewed matrix count changed')
    return ROOT / 'plan.json',plan


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', choices=('a', 'b', 'c', 'd'))
    p.add_argument('--formal', action='store_true')
    p.add_argument('--preview', action='store_true', help='Only print reviewed matrix assignments; no plan file written')
    args = p.parse_args()
    if bool(args.smoke) == bool(args.formal):
        p.error('Choose exactly one of --smoke HOST and --formal')
    path, plan = formal() if args.formal else smoke(args.smoke)
    if args.preview:
        print(json.dumps({'seed0_count':len(plan['seed0_trial_ids']),
              'lanes':[{**l,'trial_count':len(l['trial_ids'])} for l in plan['lanes']],
              'external_assignments':[{k:t[k] for k in ('id','dataset','assigned_host_key')} for t in plan['trials'] if t.get('runner_kind')=='deferred_detection']}))
        return
    if args.formal:
        for name in ('adopt_receipt.py','campaign_dispatch.py','deferred_data.py','verify_receipts.py'):
            if not (ROOT / name).is_file():
                raise FileNotFoundError(ROOT / name)
    frozen(path, plan)
    load_plan(path)
    print(json.dumps({'path': str(path), 'plan_sha256': digest(plan),
                      'trials': [t['id'] for t in plan['trials']], 'source_files': len(plan['code_hashes'])}))


if __name__ == '__main__':
    main()
