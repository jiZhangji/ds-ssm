#!/usr/bin/env python3
"""Four-model SSDD seed0 trainer, new 928/232 TEST-as-VAL protocol.

Uses the hash-bound original training framework with a small subclass replacing
implicit optimizer/accumulation/warmup behavior. No automatic resume or install.
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
import sys
from datetime import datetime, timezone

from probe_training import sha, put, sha_tensor, compute_pids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--legacy-gpu-lock', type=Path, required=True)
    p.add_argument('--variant', choices=['cnn_yolov8n', 'original', 'matched_scale075', 'full'], required=True)
    p.add_argument('--batch', type=int, choices=[4, 16], required=True)
    p.add_argument('--smoke-epochs', type=int, choices=[1, 2], help='Explicit non-paper smoke; default is formal 300e')
    a = p.parse_args()
    if os.name != 'posix' or os.environ.get('CUDA_VISIBLE_DEVICES') != a.gpu_uuid:
        raise RuntimeError('Explicit Linux server GPU UUID required')
    if a.output.exists():
        raise FileExistsError('No overwrite/resume: ' + str(a.output))
    a.output.mkdir(parents=True)
    handles = []
    locks = [a.output.parent / (a.gpu_uuid + '.training.lock')]
    if a.legacy_gpu_lock.exists():
        locks.insert(0, a.legacy_gpu_lock)
    for path in locks:
        h = path.open('a+')
        fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handles.append(h)
    if compute_pids(a.gpu_uuid):
        raise RuntimeError('GPU occupied; no takeover')
    here = Path(__file__).resolve().parent
    expected = json.loads((here / 'frozen_source_hashes.json').read_text())
    for row in expected:
        if sha(a.source_root / row['path']) != row['sha256']:
            raise RuntimeError('Frozen code mismatch: ' + row['path'])
    # The official test split is deliberately and explicitly the validation
    # selection split under the user's selected historical protocol.
    records = []
    for source_split, target_split, count in [('train', 'train', 928), ('test', 'val', 232)]:
        images = a.dataset_root / 'images' / source_split
        labels = a.dataset_root / 'labels' / source_split
        out_images = a.output / 'data/images' / target_split
        out_labels = a.output / 'data/labels' / target_split
        out_images.mkdir(parents=True)
        out_labels.mkdir(parents=True)
        found = []
        for image in sorted(images.iterdir()):
            if image.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}:
                continue
            label = labels / (image.stem + '.txt')
            for line in label.read_text().splitlines():
                values = line.split()
                if values and (len(values) != 5 or float(values[0]) != 0):
                    raise RuntimeError('Invalid single-class YOLO annotation: ' + str(label))
            found.append({'source_split': source_split, 'selection_split': target_split,
                          'name': image.name, 'image_sha256': sha(image), 'label_sha256': sha(label)})
            (out_images / image.name).symlink_to(image.resolve())
            (out_labels / label.name).symlink_to(label.resolve())
        if len(found) != count:
            raise RuntimeError(f'{source_split}: expected {count}, got {len(found)}')
        records.extend(found)
    put(a.output / 'data_manifest.json', records)
    data = {'path': str((a.output / 'data').resolve()), 'train': 'images/train',
            'val': 'images/val', 'test': 'images/val', 'names': {0: 'ship'}}
    put(a.output / 'data.yaml', data)
    sys.path.insert(0, str(a.source_root.resolve()))
    import numpy as np
    import torch
    from cuda_uuid_compat import ensure_uuid_cuda
    identity = ensure_uuid_cuda(a.gpu_uuid)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    from frozen_config_source import config_payload
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.nn.tasks import DetectionModel
    sys.path.insert(0, str(a.source_root / 'tools'))
    from validate_rgsp_expanded import audit_controls
    config = config_payload({'control': a.variant})
    put(a.output / 'model.yaml', config)
    protocol = {'schema': 'rgsp-ssdd-table-completion-v1', 'variant': a.variant, 'seed': 0,
        'split': '928 train / official232 TEST-as-VAL; no independent test claim',
        'selection': 'EMA best 0.1*AP50+0.9*AP50:95; earliest tie', 'epochs': 300,
        'physical_batch': a.batch, 'accumulation': 32 // a.batch, 'effective_batch': 32,
        'loss_normalization': 'native summed microbatch loss; divide accumulated gradients by actual samples',
        'optimizer': 'AdamW', 'lr0': 0.001, 'betas': [0.9, 0.999], 'eps': 1e-8,
        'weight_decay': {'weights': 0.0005, 'bias_and_norm': 0.0},
        'schedule': '5 full epochs linear warmup at optimizer updates, then cosine to lr0*0.01',
        'module_warmup': 0, 'early_stopping': False, 'input': [3, 640, 640],
        'precision': 'FP32 training/checkpoints/evaluation; AMP and TF32 off',
        'augmentation': 'letterbox640; horizontal+vertical flip p0.5; all other generic augmentations off',
        'detection_loss': {'box': 7.5, 'cls': 0.5, 'dfl': 1.5}, 'ema': 'original ModelEMA policy',
        'gradient_clip_l2': 10.0, 'evaluation': {'conf': 0.001, 'iou_nms': 0.7, 'max_det': 300,
                                              'P_R': 'maximum-F1 curve summary; not fixed deployment threshold'},
        'smoke_only': bool(a.smoke_epochs), 'code_sha256': sha(__file__),
        'source_manifest_sha256': sha(here / 'frozen_source_hashes.json'),
        'data_manifest_sha256': sha(a.output / 'data_manifest.json'), 'identity': identity}
    put(a.output / 'protocol.json', protocol)

    class ExplicitTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            if weights is not None:
                raise RuntimeError('Scratch only')
            random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
            model = DetectionModel(config, nc=1, verbose=False)
            paired = None
            if a.variant != 'cnn_yolov8n':
                random.seed(0); np.random.seed(0); torch.manual_seed(0)
                ref = DetectionModel(config_payload({'control': 'original'}), nc=1, verbose=False)
                state, reference = model.state_dict(), ref.state_dict()
                keys = [k for k in reference if not k.startswith('model.3.')]
                for key in keys:
                    if state[key].shape != reference[key].shape:
                        raise RuntimeError('Common tensor mismatch: ' + key)
                    state[key].copy_(reference[key])
                paired = hashlib.sha256(''.join(k + sha_tensor(state[k]) for k in sorted(keys)).encode()).hexdigest()
                del ref, state, reference
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.reset_running_stats()
            self.common_tensor_sha256 = paired
            return model

        def build_optimizer(self, model, name='auto', lr=0.001, momentum=0.9, decay=0.0005, iterations=1):
            if name != 'AdamW' or lr != 0.001 or abs(decay - 0.0005) > 1e-12:
                raise RuntimeError('Unexpected optimizer or implicit LR/WD scaling')
            groups = {'decay': [], 'norm': [], 'bias': []}
            norms = tuple(v for k, v in torch.nn.__dict__.items() if 'Norm' in k and isinstance(v, type))
            for module in model.modules():
                for key, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        groups['bias' if key == 'bias' else ('norm' if isinstance(module, norms) else 'decay')].append(param)
            return torch.optim.AdamW([{'params': params, 'weight_decay': 0.0005 if key == 'decay' else 0.0,
                                      'group_name': key} for key, params in groups.items() if params],
                                      lr=0.001, betas=(0.9, 0.999), eps=1e-8)

        def _setup_scheduler(self):
            # Framework epoch calls become identity; real schedule is set before
            # every batch from completed optimizer-update count below.
            self.lf = lambda epoch: 1.0
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.lf)

        def _setup_train(self, world_size):
            super()._setup_train(world_size)
            if world_size != 1 or self.accumulate != 32 // a.batch:
                raise RuntimeError('Single GPU fixed accumulation required')
            if len(self.train_loader.dataset) != 928 or len(self.test_loader.dataset) != 232:
                raise RuntimeError('Runtime data counts differ')
            if len(self.train_loader) % self.accumulate:
                raise RuntimeError('This frozen protocol requires complete 32-sample accumulation groups')
            self.completed_updates = 0
            self.accumulated_samples = 0
            self.saved_best = -math.inf
            self.saved_best_epoch = None
            self.last_norm = None
            put(a.output / 'runtime_audit.json', {
                'controls': audit_controls(self.model, self.optimizer), 'common_tensor_sha256': self.common_tensor_sha256,
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'accumulation': self.accumulate, 'train_batches': len(self.train_loader),
                'optimizer_groups': [{k: g[k] for k in ['group_name', 'lr', 'weight_decay', 'betas', 'eps']}
                                     for g in self.optimizer.param_groups], 'initialization': 'scratch, common layers matched, BN reset'})

        def preprocess_batch(self, batch):
            update = self.completed_updates
            warm = 5 * 29
            total = 300 * 29
            factor = ((update + 1) / warm if update < warm else
                      0.01 + 0.99 * (1 + math.cos(math.pi * (update - warm + 1) / (total - warm))) / 2)
            for group in self.optimizer.param_groups:
                group['lr'] = 0.001 * factor
            batch = super().preprocess_batch(batch)
            self.accumulated_samples += batch['img'].shape[0]
            return batch

        def optimizer_step(self):
            if self.accumulated_samples != 32:
                raise RuntimeError('Incomplete accumulation group: ' + str(self.accumulated_samples))
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.div_(self.accumulated_samples)
                    if not torch.isfinite(param.grad).all():
                        raise FloatingPointError('Nonfinite gradient')
            super().optimizer_step()
            self.completed_updates += 1
            self.accumulated_samples = 0

        def save_model(self):
            if self.accumulated_samples != 0 or self.completed_updates != (self.epoch + 1) * 29:
                raise RuntimeError('Optimizer-update ledger differs from 29 updates/epoch')
            audit_controls(self.model, self.optimizer)
            ckpt = {'epoch': self.epoch, 'best_fitness': self.best_fitness,
                    'model': copy.deepcopy(self.model).float().cpu(), 'ema': copy.deepcopy(self.ema.ema).float().cpu(),
                    'updates': self.ema.updates, 'optimizer': self.optimizer.state_dict(),
                    'train_args': vars(self.args), 'train_metrics': self.metrics,
                    'date': datetime.now(timezone.utc).isoformat(), 'version': '8.2.29',
                    'protocol_sha256': sha(a.output / 'protocol.json'), 'completed_updates': self.completed_updates}
            def save(path):
                temporary = path.with_name(path.name + '.tmp')
                torch.save(ckpt, temporary)
                os.replace(temporary, path)
            save(self.last)
            if float(self.fitness) > self.saved_best:
                self.saved_best = float(self.fitness)
                self.saved_best_epoch = self.epoch + 1
                save(self.best)
            record = {'epoch': self.epoch + 1, 'optimizer_updates': self.completed_updates,
                      'lr': [g['lr'] for g in self.optimizer.param_groups],
                      'fitness': float(self.fitness), 'best_epoch': self.saved_best_epoch,
                      'controls': audit_controls(self.model, self.optimizer)}
            with (a.output / 'optimizer_ledger.jsonl').open('a') as f:
                f.write(json.dumps(record, allow_nan=False) + '\n')
            print(json.dumps(record, allow_nan=False), flush=True)

        def final_eval(self):
            # Do not call framework strip_optimizer: it casts checkpoints to FP16.
            expected_epochs = a.smoke_epochs or 300
            if self.epoch + 1 != expected_epochs or self.completed_updates != expected_epochs * 29:
                raise RuntimeError('Refusing completion before the entire frozen update budget')
            saved = torch.load(self.best, map_location='cpu')
            if any(p.is_floating_point() and p.dtype != torch.float32 for p in saved['ema'].parameters()):
                raise RuntimeError('Serialized best EMA is not FP32')
            saved_controls = audit_controls(saved['ema'])
            del saved
            self.validator.args.plots = False
            self.validator.args.half = False
            self.metrics = self.validator(model=str(self.best))
            box = self.validator.metrics.box
            result = {'status': 'SMOKE_PASS' if a.smoke_epochs else 'TRAINING_COMPLETE',
                      'paper_eligible': not bool(a.smoke_epochs), 'protocol': protocol,
                      'epoch': self.epoch + 1, 'best_epoch': self.saved_best_epoch,
                      'optimizer_updates': self.completed_updates, 'best_sha256': sha(self.best),
                      'serialized_best_fp32_verified': True, 'serialized_best_controls': saved_controls,
                      'precision': float(box.mp) * 100, 'recall': float(box.mr) * 100,
                      'AP50': float(box.map50) * 100,
                      'AP75': float(box.all_ap[:, 5].mean()) * 100 if len(box.all_ap) else 0.0,
                      'AP50_95': float(box.map) * 100,
                      'selection_warning': 'Official test used for repeated validation/selection; not independent test'}
            put(a.output / 'result.json', result)

    opts = dict(model=str(a.output / 'model.yaml'), data=str(a.output / 'data.yaml'),
        project=str(a.output), name='run', exist_ok=True, pretrained=False, resume=False,
        device='', epochs=a.smoke_epochs or 300, patience=0, imgsz=640, batch=a.batch,
        workers=4, seed=0, deterministic=True, amp=False, half=False, cache=False,
        optimizer='AdamW', lr0=0.001, lrf=0.01, momentum=0.9, weight_decay=0.0005,
        nbs=32, warmup_epochs=0.0, warmup_bias_lr=0.0, warmup_momentum=0.9,
        module_warmup_epochs=0, cos_lr=False, lr_schedule='fixed', multi_scale=False,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, mosaic=0.0, close_mosaic=0,
        degrees=0.0, translate=0.0, scale=0.0, flipud=0.5, fliplr=0.5,
        shear=0.0, perspective=0.0, mixup=0.0, copy_paste=0.0,
        box=7.5, cls=0.5, dfl=1.5, val=True, split='val', conf=0.001, iou=0.7,
        max_det=300, plots=False, save=True, save_period=-1, save_json=True)
    trainer = ExplicitTrainer(overrides=opts)
    trainer.callbacks = {}  # No integrations/network telemetry; all audits explicit.
    trainer.train()


if __name__ == '__main__':
    main()
