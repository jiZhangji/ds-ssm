#!/usr/bin/env python3
"""Server-only training-step audit. Train images only; never paper results.

Uses frozen network/loss/data-loader implementations, explicit FP32 AdamW,
fixed accumulation, sample-normalized gradients and paired common tensors.
No official val/test images or labels are enumerated or loaded.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def put(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def compute_pids(uuid):
    p = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    return [int(r.split(',')[1]) for r in p.stdout.splitlines() if r.split(',')[0].strip() == uuid]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', required=True, type=Path)
    p.add_argument('--train-images', required=True, type=Path)
    p.add_argument('--train-labels', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--legacy-gpu-lock', required=True, type=Path)
    p.add_argument('--variant', choices=['cnn_yolov8n', 'original', 'matched_scale075', 'full'], required=True)
    p.add_argument('--batch', choices=[4, 16], type=int, required=True)
    p.add_argument('--updates', type=int, default=20)
    p.add_argument('--warmup-updates', type=int, default=5)
    args = p.parse_args()
    if args.updates < 10 or args.warmup_updates < 2:
        p.error('Require >=10 measured and >=2 warmup optimizer updates')
    if os.name != 'posix' or os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu_uuid:
        raise RuntimeError('Linux server and exact assigned CUDA UUID are required')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    handles = []
    # Do not create files in an old campaign. Respect its GPU lock if it exists.
    paths = [args.output.parent / (args.gpu_uuid + '.training.lock')]
    if args.legacy_gpu_lock.exists():
        paths.insert(0, args.legacy_gpu_lock)
    for path in paths:
        handle = path.open('a+')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handles.append(handle)
    if compute_pids(args.gpu_uuid):
        raise RuntimeError('GPU has an existing compute process; no takeover')
    here = Path(__file__).resolve().parent
    expected = json.loads((here / 'frozen_source_hashes.json').read_text())
    mismatches = [r['path'] for r in expected if not (args.source_root / r['path']).is_file()
                  or sha(args.source_root / r['path']) != r['sha256']]
    if mismatches:
        raise RuntimeError('Frozen source differs: ' + repr(mismatches[:12]))
    # Bind every source train image+label; use task-owned symlinks so cache files
    # are written only under this new run directory.
    staged_images = args.output / 'data/images/train'
    staged_labels = args.output / 'data/labels/train'
    staged_images.mkdir(parents=True)
    staged_labels.mkdir(parents=True)
    rows = []
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    for im in sorted(args.train_images.iterdir()):
        if im.suffix.lower() not in exts:
            continue
        label = args.train_labels / (im.stem + '.txt')
        if not label.is_file():
            raise FileNotFoundError(label)
        for line in label.read_text().splitlines():
            if line.strip() and (len(line.split()) != 5 or float(line.split()[0]) != 0):
                raise RuntimeError('Expected single-class YOLO train labels: ' + str(label))
        rows.append({'image': str(im.resolve()), 'image_sha256': sha(im),
                     'label': str(label.resolve()), 'label_sha256': sha(label)})
        (staged_images / im.name).symlink_to(im.resolve())
        (staged_labels / label.name).symlink_to(label.resolve())
    if not rows:
        raise RuntimeError('No explicitly supplied training images')
    put(args.output / 'train_manifest.json', rows)
    sys.path.insert(0, str(args.source_root.resolve()))
    import numpy as np
    import torch
    from cuda_uuid_compat import ensure_uuid_cuda
    identity = ensure_uuid_cuda(args.gpu_uuid)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from frozen_config_source import config_payload
    sys.path.insert(0, str(args.source_root / 'tools'))
    from validate_rgsp_expanded import audit_controls

    def seed():
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    def make(variant):
        payload = config_payload({'control': variant})
        config = args.output / (variant + '.yaml')
        if not config.exists():
            put(config, payload)  # JSON is valid YAML.
        seed()
        return YOLO(str(config)).model

    model = make(args.variant)
    paired = None
    if args.variant != 'cnn_yolov8n':
        reference = make('original') if args.variant != 'original' else copy.deepcopy(model)
        state = model.state_dict()
        ref = reference.state_dict()
        shared = [k for k in ref if not k.startswith('model.3.')]
        if any(k not in state or state[k].shape != ref[k].shape for k in shared):
            raise RuntimeError('Common tensor mapping mismatch')
        for k in shared:
            state[k].copy_(ref[k])
        paired = hashlib.sha256(''.join(k + sha_tensor(state[k]) for k in sorted(shared)).encode()).hexdigest()
        del reference, state, ref
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
    cfg = get_cfg(overrides=dict(task='detect', mode='train', imgsz=640, batch=args.batch,
        workers=4, amp=False, cache=False, rect=False, seed=0, deterministic=True,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0,
        shear=0.0, perspective=0.0, flipud=0.5, fliplr=0.5, mosaic=0.0,
        mixup=0.0, copy_paste=0.0, close_mosaic=0, multi_scale=False,
        box=7.5, cls=0.5, dfl=1.5))
    model.args = cfg
    model.nc = 1
    model.names = {0: 'ship'}
    model = model.cuda().train()
    controls = audit_controls(model)
    if len(controls) != (0 if args.variant in {'original', 'cnn_yolov8n'} else 1):
        raise RuntimeError('Unexpected number of controlled blocks')
    groups = {'decay': [], 'norm': [], 'bias': []}
    norm_types = tuple(v for k, v in torch.nn.__dict__.items() if 'Norm' in k and isinstance(v, type))
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            group = 'bias' if name == 'bias' else ('norm' if isinstance(module, norm_types) else 'decay')
            groups[group].append(param)
    optimizer = torch.optim.AdamW([
        {'params': params, 'weight_decay': 0.0005 if name == 'decay' else 0.0, 'group_name': name}
        for name, params in groups.items() if params], lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    initial = {'identity': identity, 'hostname': socket.gethostname(), 'variant': args.variant,
               'source_files_verified': len(expected), 'source_manifest_sha256': sha(here / 'frozen_source_hashes.json'),
               'train_manifest_sha256': sha(args.output / 'train_manifest.json'), 'train_count': len(rows),
               'physical_batch': args.batch, 'accumulation': 32 // args.batch, 'effective_batch': 32,
               'precision': 'FP32, AMP and TF32 off', 'seed': 0, 'common_tensor_sha256': paired,
               'controls': controls, 'parameters': sum(p.numel() for p in model.parameters()),
               'optimizer': [{'name': g['group_name'], 'lr': g['lr'], 'weight_decay': g['weight_decay'],
                              'betas': g['betas'], 'eps': g['eps']} for g in optimizer.param_groups],
               'probe_only': True, 'paper_eligible': False, 'heldout_labels_read': False,
               'note': 'Train-step capacity/throughput probe; fixed lr1e-3, not a completed training schedule.'}
    put(args.output / 'initial_audit.json', initial)
    dataset = build_yolo_dataset(cfg, str(staged_images), args.batch, {'nc': 1, 'names': {0: 'ship'}},
                                 mode='train', rect=False, stride=32)
    loader = build_dataloader(dataset, args.batch, 4, shuffle=True, rank=-1)
    iterator = iter(loader)
    torch.cuda.reset_peak_memory_stats()
    timings, losses = [], []
    for update in range(args.warmup_updates + args.updates):
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        samples = 0
        batch_losses = []
        for micro in range(32 // args.batch):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch['img'] = batch['img'].cuda(non_blocking=True).float() / 255.0
            n = batch['img'].shape[0]
            loss, items = model(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()  # Native detection loss is microbatch_mean * n.
            samples += n
            batch_losses.append([float(v) for v in items.detach().cpu()])
        for param in model.parameters():
            if param.grad is not None:
                param.grad.div_(samples)
                if not torch.isfinite(param.grad).all():
                    raise FloatingPointError('Nonfinite gradient')
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
        if not math.isfinite(grad_norm) or grad_norm <= 0:
            raise FloatingPointError('Invalid gradient norm')
        optimizer.step()
        audit_controls(model, optimizer)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        record = {'update': update + 1, 'seconds': elapsed, 'samples': samples,
                  'gradient_norm_before_clip': grad_norm, 'loss_items': batch_losses,
                  'measured': update >= args.warmup_updates}
        with (args.output / 'updates.jsonl').open('a') as f:
            f.write(json.dumps(record, allow_nan=False) + '\n')
        if record['measured']:
            timings.append(elapsed)
            losses.extend(batch_losses)
        print(json.dumps({k: record[k] for k in ['update', 'seconds', 'samples', 'measured']}), flush=True)
    result = {**initial, 'status': 'PROBE_PASS', 'measured_updates': len(timings),
              'mean_update_seconds': sum(timings) / len(timings),
              'images_per_second': 32 * len(timings) / sum(timings),
              'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
              'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3,
              'final_controls': audit_controls(model, optimizer), 'finite_gradient_audit': True,
              'estimated_ssdd_300e_training_hours_excluding_validation': 29 * 300 * sum(timings) / len(timings) / 3600,
              'last_loss_items': losses[-1], 'exclusive_gpu_pids': compute_pids(args.gpu_uuid)}
    if set(result['exclusive_gpu_pids']) != {os.getpid()}:
        raise RuntimeError('Concurrent GPU process discovered')
    put(args.output / 'result.json', result)
    print(json.dumps(result, allow_nan=False), flush=True)


def sha_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


if __name__ == '__main__':
    main()
