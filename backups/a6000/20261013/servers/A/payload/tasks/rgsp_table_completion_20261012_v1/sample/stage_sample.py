#!/usr/bin/env python3
"""Independent SAMPLE full-byte/pixel audit and immutable source/data staging.

Run only on assigned A server. No torch/model imports or target-domain fitting.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess

GPU = 'GPU-12cf4841-46b9-94d1-2715-9932134cf964'
HOST = 'powerleader-PT6620W'
ROOT = Path('/home/powerleader/ig-ds-ssm/tasks/rgsp_table_completion_20261012_v1/sample')
SOURCE = Path('/home/powerleader/ig-ds-ssm/data/cross_task_prepared_v2/classification/SAMPLE/_raw/SAMPLE_dataset_public/decibel')
OLD = Path('/home/powerleader/ig-ds-ssm/tasks/rgsp_seedwise_20261009_v1')
CLASSES = ['2s1', 'bmp2', 'btr70', 'm1', 'm2', 'm35', 'm548', 'm60', 't72', 'zsu23']
EXPECTED = {'2s1': (116,58), 'bmp2': (55,52), 'btr70': (43,49), 'm1': (78,51), 'm2': (75,53), 'm35': (76,53), 'm548': (75,53), 'm60': (116,60), 't72': (56,52), 'zsu23': (116,58)}

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write('\n')

def assert_idle():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=15)
    if any(line.startswith(GPU) for line in result.stdout.splitlines()):
        raise RuntimeError('Assigned GPU has an existing compute process')
    return result.stdout.strip()

def run():
    if os.name != 'posix' or socket.gethostname() != HOST or Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('Staging pinned to assigned A and independent sample root')
    from PIL import Image
    started = datetime.now(timezone.utc).isoformat()
    (ROOT / 'locks').mkdir(exist_ok=True)
    with ExitStack() as stack:
        legacy = OLD / 'locks' / (GPU + '.lock')
        if not legacy.is_file():
            raise RuntimeError('Expected pre-existing legacy lock absent')
        for path, mode in [(legacy, 'r+'), (ROOT / 'locks' / (GPU + '.lock'), 'a+')]:
            handle = stack.enter_context(path.open(mode))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_idle()
        data, code = ROOT / 'data', ROOT / 'source'
        if data.exists() or code.exists():
            raise FileExistsError('Never overwrite a staged dataset/source; use an explicitly new revision')
        data.mkdir()
        code.mkdir()
        all_rows, records = [], []
        for path in sorted(SOURCE.glob('*/*/*.png')):
            relative = path.relative_to(SOURCE).as_posix()
            match = re.fullmatch(r'(real|synth)/([^/]+)/([^/]+)_(real|synth)_A_elevDeg_(\d{3})_azCenter_(\d+)_(\d+)_serial_([^/]+)\.png', relative)
            if not match:
                raise RuntimeError('Unrecognized SAMPLE filename: ' + relative)
            domain, category, named_category, named_domain, angle, az_whole, az_decimal, serial = match.groups()
            if category != named_category or domain != named_domain or category not in CLASSES or int(angle) not in (14,15,16,17):
                raise RuntimeError('Domain/class/elevation metadata conflict: ' + relative)
            if path.is_symlink() or not path.resolve().is_relative_to(SOURCE.resolve()):
                raise RuntimeError('Unexpected source symlink or path escape')
            before = sha(path)
            with Image.open(path) as image:
                image.load()
                rgb = image.convert('RGB')
                width, height = rgb.size
                pixel = hashlib.sha256(canonical({'mode':'RGB', 'width':width, 'height':height}) + rgb.tobytes()).hexdigest()
                mode = image.mode
            if sha(path) != before:
                raise RuntimeError('Source changed while decoding')
            split = 'train' if domain == 'synth' and int(angle) in (14,15,16) else 'test' if domain == 'real' and int(angle) == 17 else 'excluded'
            row = {'id': relative, 'relative_path': relative, 'source_path': str(path), 'domain': domain, 'class_name': category, 'label': CLASSES.index(category), 'elevation': int(angle), 'azimuth': az_whole + '.' + az_decimal, 'serial': serial, 'split': split, 'byte_sha256': before, 'pixel_sha256': pixel, 'width': width, 'height': height, 'source_mode': mode, 'bytes':path.stat().st_size}
            all_rows.append(row)
            if split != 'excluded':
                destination = data / 'files' / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with path.open('rb') as src, destination.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
                if sha(destination) != before:
                    raise RuntimeError('Staged byte mismatch')
                records.append(dict(row, path=str(destination)))
        observed = Counter((r['domain'],r['elevation']) for r in all_rows)
        expected_angles = {(d,a):n for d in ('real','synth') for a,n in ((14,96),(15,197),(16,513),(17,539))}
        if len(all_rows) != 2690 or observed != expected_angles:
            raise RuntimeError('Source counts differ from independently observed full2690 protocol')
        counts = Counter(r['split'] for r in records)
        if counts != {'train':806, 'test':539}:
            raise RuntimeError('Selected counts differ from 806/539')
        for c,(train,test) in EXPECTED.items():
            if Counter(r['split'] for r in records if r['class_name']==c) != {'train':train,'test':test}:
                raise RuntimeError('Unexpected class split counts: ' + c)
        duplicates = {}
        for key in ('byte_sha256','pixel_sha256'):
            groups = defaultdict(list)
            for row in records:
                groups[row[key]].append(row)
            duplicates[key] = [[r['id'] for r in group] for group in groups.values() if len(group)>1]
            for group in groups.values():
                if len({r['split'] for r in group})>1 or len({r['label'] for r in group})>1:
                    raise RuntimeError('Cross-split or class-conflicting exact content duplicate')
        serials = {c: sorted({r['serial'] for r in all_rows if r['class_name']==c}) for c in CLASSES}
        if any(len(v)!=1 for v in serials.values()):
            raise RuntimeError('Serial inventory changed from source audit')
        common = {'schema':'sample-domain-angle-v1', 'task':'classification', 'dataset':'SAMPLE', 'classes':CLASSES, 'protocol':'synth14-16_to_real17_fixed10000', 'source_root':str(SOURCE)}
        write(data/'manifest.json', dict(common, records=records))
        write(data/'manifest_train.json', dict(common, records=[r for r in records if r['split']=='train']))
        write(data/'manifest_eval.json', dict(common, records=[r for r in records if r['split']=='test']))
        write(data/'source_inventory.json', {'schema':'sample-source-inventory-v1', 'records':all_rows})
        plan = json.loads((OLD/'plan.json').read_text())
        old_code = OLD / 'code'
        frozen = []
        for entry in plan['code_hashes']:
            original = Path(entry['path'])
            if not original.is_relative_to(old_code):
                continue
            if not original.is_file() or original.is_symlink() or sha(original) != entry['sha256']:
                raise RuntimeError('Frozen source mismatch: ' + str(original))
            rel = original.relative_to(old_code)
            destination = code / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            with original.open('rb') as src, destination.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            if sha(destination) != entry['sha256']:
                raise RuntimeError('Source copy mismatch')
            frozen.append({'path':rel.as_posix(),'sha256':entry['sha256'],'original_path':str(original)})
        if not frozen or not (code/'tools/run_shared_sar_tasks_v2.py').is_file():
            raise RuntimeError('Frozen source incomplete')
        write(ROOT/'source_inventory.json', {'source_root':str(code), 'copied_from':str(old_code), 'old_plan_sha256':sha(OLD/'plan.json'), 'files':frozen})
        audit = {'schema':'sample-source-data-audit-v1', 'status':'PASS', 'started_utc':started, 'finished_utc':datetime.now(timezone.utc).isoformat(), 'hostname':socket.gethostname(), 'source_root':str(SOURCE), 'all_source_count':2690, 'selected_count':1345, 'counts':dict(counts), 'all_source_domain_elevation_counts':[{'domain':d,'elevation':a,'count':n} for (d,a),n in sorted(observed.items())], 'per_class':{c:{'train':a,'test':b} for c,(a,b) in EXPECTED.items()}, 'serials':serials, 'instance_or_serial_disjoint':False, 'domain_and_elevation_disjoint':True, 'selected_content_duplicates':duplicates, 'pixel_hash_definition':'sha256(canonical_json_RGB_width_height + decoded_RGB_bytes)', 'all_source_decoded':True, 'source_images_modified':False, 'test_domain_use':'integrity audit only; eval manifest is separate and forbidden to probe/training loader', 'source_inventory_sha256':sha(data/'source_inventory.json'), 'manifest_sha256':sha(data/'manifest.json'), 'manifest_train_sha256':sha(data/'manifest_train.json'), 'manifest_eval_sha256':sha(data/'manifest_eval.json'), 'source_code_inventory_sha256':sha(ROOT/'source_inventory.json'), 'source_code_files':len(frozen), 'legacy_lock':str(legacy), 'new_lock':str(ROOT/'locks'/(GPU+'.lock')), 'gpu_compute_processes':assert_idle()}
        write(data/'audit.json',audit)
        print(json.dumps({k:audit[k] for k in ('status','all_source_count','counts','selected_content_duplicates','source_code_files','manifest_train_sha256','manifest_eval_sha256')},ensure_ascii=False),flush=True)

if __name__ == '__main__':
    run()
