#!/usr/bin/env python3
"""Isolated, GPU-idle-gated FMP-inspired AIR experiment; never interrupts jobs."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CODE = ROOT / 'code'
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / 'tools'))

def save(name, value):
    value['timestamp'] = datetime.datetime.now().astimezone().isoformat()
    path = ROOT / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def prepare():
    from prepare_cross_task_v2 import extract_once, airship_annotations, tile_detection_record
    from PIL import Image
    import yaml
    incoming = ROOT / 'incoming'
    data = extract_once(incoming / 'AIR-SARShip-2.0-data.zip', ROOT / 'raw_images')
    xml = extract_once(incoming / 'AIR-SARShip-2.0-xml.zip', ROOT / 'raw_xml')
    xml_root = next(p.parent for p in xml.rglob('*.xml'))
    image_root = next(p.parent for p in data.rglob('*.tiff'))
    records = airship_annotations(xml_root, image_root)
    assert len(records) == 300, len(records)
    random.Random(0).shuffle(records)
    parts = {'train': records[:240], 'test': records[240:]}
    out = ROOT / 'dataset'
    out.mkdir(exist_ok=False)
    manifest = {'seed': 0, 'split_before_tiling': True, 'tile': 512, 'stride': 256,
                'keep_empty': True, 'minimum_box_retained_fraction': 0.5,
                'image_conversion': 'PIL RGB', 'source_zip_sha256': {}, 'splits': {},
                'image_sizes': sorted(set(r['size'] for r in records)), 'source_samples': []}
    for p in sorted(incoming.glob('*.zip')):
        manifest['source_zip_sha256'][p.name] = digest(p)
    for r in records[:5]:
        with Image.open(r['image']) as im:
            manifest['source_samples'].append({'name': r['image'].name, 'mode': im.mode,
                                               'size': im.size, 'extrema': im.getextrema()})
    for split, items in parts.items():
        counts = {'source_ids': [r['image'].stem for r in items], 'source_images': len(items),
                  'source_objects': sum(len(r['boxes']) for r in items),
                  'tiles': 0, 'tile_objects': 0, 'empty_tiles': 0}
        for r in items:
            tiles, objects = tile_detection_record(r, out, split, tile=512, stride=256, keep_empty=True)
            counts['tiles'] += tiles
            counts['tile_objects'] += objects
        labels = list((out / 'labels' / split).glob('*.txt'))
        counts['empty_tiles'] = sum(not p.read_text().strip() for p in labels)
        assert len(labels) == counts['tiles'] and counts['tiles'] > 0 and counts['tile_objects'] > 0
        counts['labels_sha256'] = hashlib.sha256(''.join(p.name + digest(p) for p in sorted(labels)).encode()).hexdigest()
        manifest['splits'][split] = counts
    assert not set(manifest['splits']['train']['source_ids']) & set(manifest['splits']['test']['source_ids'])
    (out / 'AIR.yaml').write_text(yaml.safe_dump({'path': str(out), 'train': 'images/train',
                                                 'val': 'images/test', 'test': 'images/test',
                                                 'names': {0: 'ship'}}, sort_keys=False))
    save('data_manifest.json', manifest)
    sources = {str(p.relative_to(CODE)): digest(p) for p in sorted(CODE.rglob('*'))
               if p.is_file() and p.suffix in ('.py', '.yaml', '.yml', '.toml')}
    save('code_manifest.json', {'sha256': sources})
    save('status.json', {'state': 'PREPARED', 'queue_pid': os.getpid()})

def wait_for_gpu():
    stable = 0
    while stable < 2:
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                              '--format=csv,noheader'], text=True).strip()
        util = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=utilization.gpu',
                                           '--format=csv,noheader,nounits'], text=True).strip())
        free_mib = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free',
                                               '--format=csv,noheader,nounits'], text=True).strip())
        other_processes = '\n'.join(line for line in processes.splitlines()
                                    if line.split(',')[0].strip() != str(os.getpid()))
        available = not other_processes and util <= 5 and free_mib >= 40000
        stable = stable + 1 if available else 0
        save('status.json', {'state': 'WAITING_FOR_GPU', 'queue_pid': os.getpid(),
                             'compute_processes': other_processes, 'gpu_utilization': util,
                             'free_mib': free_mib, 'idle_checks': stable})
        print('GPU gate:', processes or 'no compute processes', 'util=', util, flush=True)
        if stable < 2:
            time.sleep(60)

def train_all():
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    import torch

    class FixedLastTrainer(DetectionTrainer):
        # Keep the held-out test entirely out of checkpoint selection.
        def validate(self):
            return dict(self.metrics), 0.0

        def final_eval(self):
            pass

    for name, config in [('baseline', 'air_sarship_20_baseline_sd16.yaml'),
                         ('rgsp', 'air_sarship_20_full_p3_sd16.yaml')]:
        wait_for_gpu()
        params = dict(data=str(ROOT / 'dataset/AIR.yaml'), epochs=300, imgsz=640, batch=32, nbs=96,
                      optimizer='Adam', lr0=0.001, lrf=0.01, momentum=0.9, weight_decay=0.0005,
                      warmup_epochs=0.0, module_warmup_epochs=0, cos_lr=True, lr_schedule='auto',
                      mosaic=1.0, close_mosaic=0, scale=0.5, translate=0.1, degrees=0.0,
                      flipud=0.5, fliplr=0.5, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                      mixup=0.0, copy_paste=0.0, shear=0.0, perspective=0.0, multi_scale=False,
                      workers=8, device='0', amp=False, cache=False, seed=0, deterministic=True,
                      pretrained=False, patience=301, val=False, plots=False, save=True, save_period=10,
                      project=str(ROOT / 'runs'), name=name, exist_ok=False)
        save(name + '_recipe.json', params)
        batches = [0]

        def start(trainer):
            assert trainer.accumulate == 3, trainer.accumulate
            save('status.json', {'state': 'TRAINING', 'variant': name, 'epoch': 0,
                                 'accumulate': trainer.accumulate, 'micro_batch': trainer.batch_size,
                                 'pid': os.getpid(), 'torch': torch.__version__})

        def batch_end(trainer):
            assert trainer.accumulate == 3
            batches[0] += 1
            if batches[0] == 3:
                save(name + '_first_three_batches.json', {'finite_loss': True, 'accumulate': 3,
                     'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 1024 ** 3,
                     'peak_reserved_GiB': torch.cuda.max_memory_reserved() / 1024 ** 3})

        def epoch_end(trainer):
            save('status.json', {'state': 'TRAINING', 'variant': name, 'epoch': trainer.epoch + 1,
                                 'epochs': 300, 'pid': os.getpid(), 'accumulate': trainer.accumulate})

        model = YOLO(str(ROOT / 'incoming' / config))
        model.add_callback('on_train_start', start)
        model.add_callback('on_train_batch_end', batch_end)
        model.add_callback('on_fit_epoch_end', epoch_end)
        model.train(trainer=FixedLastTrainer, **params)
        last = ROOT / 'runs' / name / 'weights/last.pt'
        assert last.exists()
        save('status.json', {'state': 'EVALUATING', 'variant': name, 'pid': os.getpid()})
        subprocess.run([sys.executable, str(CODE / 'tools/evaluate_sarmix.py'), '--weights', str(last),
                        '--data', str(ROOT / 'dataset/AIR.yaml'), '--experiment', 'air_fmp_recipe_20261007_c_' + name,
                        '--placement', 'baseline' if name == 'baseline' else 'p3',
                        '--variant', 'baseline' if name == 'baseline' else 'full', '--state-dim', '16',
                        '--seed', '0', '--subset', 'AIR-SARShip-2.0', '--split', 'test',
                        '--output-json', str(ROOT / (name + '_test.json')), '--imgsz', '640',
                        '--batch', '16', '--device', '0', '--workers', '8', '--iou', '0.6',
                        '--project', str(ROOT / 'evaluations'), '--name', name], cwd=CODE, check=True)
        del model
        torch.cuda.empty_cache()
    save('status.json', {'state': 'COMPLETE', 'pid': os.getpid()})

if __name__ == '__main__':
    try:
        if sys.argv[1:] == ['--prepare']:
            prepare()
        else:
            assert (ROOT / 'data_manifest.json').exists(), 'Run --prepare first'
            train_all()
    except Exception as exc:
        save('status.json', {'state': 'FAILED', 'error': repr(exc), 'pid': os.getpid()})
        raise
