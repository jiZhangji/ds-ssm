"""Exact TorchVision R50-FPN detectors under the declared SAR training recipe.

These are new, explicitly named common-protocol baselines. They do not fill
unreported columns beside scores copied from a different literature protocol.
CUDA-only; frozen source, dataset, FP32 checkpoint and all metrics are bound.
"""
import argparse
import copy
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'detection'))
import run_cross as common
from data_binding import prepare_data, recheck_data
from detector_spec import PROTOCOL, update_budget

METHODS = {'fasterrcnn_resnet50_fpn', 'retinanet_resnet50_fpn', 'fcos_resnet50_fpn'}


def make_model(method, nc):
    import torchvision
    from torchvision.models import detection
    if method not in METHODS:
        raise ValueError('Exact supported method id required')
    builder = getattr(detection, method)
    if not {'weights', 'weights_backbone'} <= set(inspect.signature(builder).parameters):
        raise RuntimeError('A supported explicit scratch-only weights API is required')
    options = {'weights': None, 'weights_backbone': None, 'num_classes': nc + 1,
               'min_size': 640, 'max_size': 640, 'image_mean': [0., 0., 0.], 'image_std': [1., 1., 1.]}
    if method.startswith('fasterrcnn'):
        options.update(box_score_thresh=0.001, box_nms_thresh=0.7, box_detections_per_img=300)
    else:
        options.update(score_thresh=0.001, nms_thresh=0.7, detections_per_img=300)
    model = builder(**options)
    metadata = {'library': 'torchvision', 'version': torchvision.__version__, 'builder': method,
                'constructor': options, 'backbone': 'ResNet50-FPN', 'weights_downloaded': False,
                'background_label': 0, 'foreground_labels': list(range(1, nc + 1)),
                'literature_scores_reproduced': False,
                'normalization': 'RGB/255; model normalization identity, matching the shared SAR input',
                'head_classes': nc + 1,
                'dense_head_note': 'RetinaNet/FCOS keep unused class0 logits with foreground at1; class0 predictions are excluded from SAR metrics.'}
    return model, metadata


def batch_to_targets(batch, device):
    import torch
    tensor = batch['img'].to(device).float() / 255.
    height, width = tensor.shape[-2:]
    targets = []
    for index in range(tensor.shape[0]):
        selected = batch['batch_idx'].view(-1) == index
        xywh = batch['bboxes'][selected].to(device)
        center = xywh[:, :2] * xywh.new_tensor([width, height])
        half = xywh[:, 2:] * xywh.new_tensor([width, height]) / 2
        boxes = torch.cat((center - half, center + half), 1)
        labels = batch['cls'][selected].view(-1).to(device=device, dtype=torch.int64) + 1
        if len(boxes) and not bool((boxes[:, 2:] > boxes[:, :2]).all()):
            raise ValueError('Degenerate target box; no silent relabeling or removal')
        targets.append({'boxes': boxes, 'labels': labels})
    return list(tensor), targets


def inventory(source_root):
    import torchvision
    paths = [Path(__file__), ROOT / 'run_cross.py', ROOT / 'cuda_uuid_compat.py']
    paths.extend((ROOT / 'detection').glob('*.py'))
    paths.extend(p for p in (source_root / 'ultralytics').rglob('*') if p.suffix in {'.py', '.yaml', '.yml'})
    library = Path(torchvision.__file__).parent
    paths.extend(p for p in (library / 'models').rglob('*.py'))
    paths.extend(p for p in (library / 'ops').rglob('*.py'))
    return [{'path': str(p), 'sha256': common.sha(p)} for p in sorted(set(paths))]


def run(args):
    if not sys.platform.startswith('linux'):
        raise RuntimeError('Registered Linux CUDA server only')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['YOLO_CONFIG_DIR'] = str(args.output / 'settings')
    os.environ['MPLCONFIGDIR'] = str(args.output / 'matplotlib')
    os.environ['WANDB_MODE'] = 'disabled'
    common.require_gpu_alone(args.gpu_uuid)
    args.output.mkdir(parents=True, exist_ok=True)
    if any(p.name != 'run.log' for p in args.output.iterdir()):
        raise FileExistsError('Preserve the previous attempt')
    data, records, audit = prepare_data(args)
    sys.path.insert(0, str(args.source_root))
    import numpy as np
    import torch
    import torchvision
    from cuda_uuid_compat import ensure_uuid_cuda
    identity = ensure_uuid_cuda(args.gpu_uuid)
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_yolo_dataset
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils.metrics import DetMetrics
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, metadata = make_model(args.method, data['nc'])
    model = model.float().cuda()
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    optimizer = common.optimizer_for(torch, model)
    budget = update_budget(audit['expected_counts']['train'])
    sources = inventory(args.source_root)
    protocol = {**copy.deepcopy(PROTOCOL), 'schema': 'rgsp-torchvision-detection-20261013-v1',
                'method': args.method, 'dataset_id': args.dataset_id, 'model': metadata,
                'source_sha256': common.digest(sources),
                'data_manifest_sha256': common.sha(args.dataset_manifest), 'budget': budget,
                'initialization': 'scratch official TorchVision constructor, no weights',
                'loss': 'sum of native TorchVision loss terms; multiplied by microbatch images then divided by actual accumulated images',
                'source_data_protocol': audit['source_data_protocol']}
    protocol_sha = common.digest(protocol)
    common.write_json(args.output / 'protocol.json', {'protocol_sha256': protocol_sha, 'protocol': protocol})
    common.write_json(args.output / 'source_inventory.json', sources)
    common.write_json(args.output / 'runtime.json', {**identity, 'torchvision': torchvision.__version__,
        'seed': args.seed, 'mode': args.mode, 'physical_batch': 16, 'accumulation': 4,
        'params': sum(p.numel() for p in model.parameters()), 'protocol_sha256': protocol_sha})
    aug = PROTOCOL['augmentation']
    cfg = get_cfg(overrides=dict(task='detect', mode='train', imgsz=640, batch=16, workers=0,
        amp=False, half=False, cache=False, rect=False, fraction=1., pretrained=False,
        epochs=150, close_mosaic=15, seed=args.seed, deterministic=True,
        hsv_h=aug['hsv_h'], hsv_s=aug['hsv_s'], hsv_v=aug['hsv_v'], degrees=aug['degrees'],
        translate=aug['translate'], scale=aug['scale'], shear=0., perspective=0.,
        flipud=.5, fliplr=.5, mosaic=.3, mixup=0., copy_paste=0.,
        conf=.001, iou=.7, max_det=300, plots=False, save_json=False))
    generator = torch.Generator().manual_seed(6148914691236517205 + args.seed)

    def loader(split):
        training = split == 'train'
        ds = build_yolo_dataset(cfg, data[split], 16, data,
            mode='train' if training else 'val', rect=not training, stride=32)
        if len(ds) != audit['expected_counts'][split]:
            raise RuntimeError('Dataset loader discarded files')
        return torch.utils.data.DataLoader(ds, batch_size=16, shuffle=training, num_workers=0,
            collate_fn=ds.collate_fn, drop_last=False, generator=generator, pin_memory=True)

    train_loader = loader('train')

    def evaluate(selected, data_loader):
        validator = DetectionValidator(args={'conf': .001, 'iou': .7, 'plots': False})
        validator.iouv = torch.linspace(.5, .95, 10, device='cuda')
        metrics = DetMetrics(names=data['names'])
        rows = []
        selected.eval()
        with torch.no_grad():
            for batch in data_loader:
                images, targets = batch_to_targets(batch, 'cuda')
                predictions = selected(images)
                for pred, target in zip(predictions, targets):
                    keep = (pred['labels'] > 0) & (pred['scores'] >= .001)
                    boxes, scores, labels = pred['boxes'][keep], pred['scores'][keep], pred['labels'][keep] - 1
                    detections = torch.cat((boxes, scores[:, None], labels[:, None]), 1)
                    correct = validator._process_batch(detections, target['boxes'], target['labels'] - 1)
                    rows.append((correct.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy(),
                                 (target['labels'] - 1).cpu().numpy()))
        arrays = [np.concatenate([r[i] for r in rows], axis=0) for i in range(4)]
        if len(arrays[1]) and len(arrays[3]):
            metrics.process(*arrays)
        box = metrics.box
        result = {'P': float(box.mp), 'R': float(box.mr), 'AP50': float(box.map50),
            'AP75': float(box.all_ap[:, 5].mean()) if len(box.all_ap) else 0., 'AP50_95': float(box.map)}
        assert all(math.isfinite(v) and 0 <= v <= 1 for v in result.values())
        return result

    val_loader = loader('val') if args.mode == 'train' else None
    start = time.perf_counter()
    updates, best_epoch, stale = 0, 0, 0
    best_score = -math.inf
    timings = []
    for epoch in range(1, 151):
        common.require_gpu_alone(args.gpu_uuid, initialized=True)
        epoch_start = time.perf_counter()
        if epoch == 136:
            train_loader.dataset.close_mosaic(hyp=cfg)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated, samples, loss_sum = 0, 0, 0.
        step_start = time.perf_counter()
        for index, batch in enumerate(train_loader):
            images, targets = batch_to_targets(batch, 'cuda')
            n = len(images)
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite native detector loss')
            (loss * n).backward()
            accumulated += n
            samples += n
            loss_sum += float(loss.detach()) * n
            if (index + 1) % 4 == 0 or index + 1 == len(train_loader):
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(accumulated)
                if updates == 0:
                    common.write_json(args.output / 'gradient_audit.json', common.basic_gradient_audit(torch, model))
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                updates += 1
                lr = common.learning_rate(updates, budget['updates_per_epoch'])
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.step()
                common.update_ema(torch, ema, model, updates)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - step_start
                timings.append(seconds)
                common.append_json(args.output / 'updates.jsonl', {'epoch': epoch, 'update': updates,
                    'samples': accumulated, 'lr': lr, 'gradient_norm': float(norm), 'seconds': seconds})
                accumulated = 0
                step_start = time.perf_counter()
                if args.mode == 'probe' and updates == 6:
                    # A real save/reload check and train-image inference, never holdout selection.
                    state = {k:v.detach().cpu().clone() for k,v in ema.state_dict().items()}
                    common.checkpoint(torch, args.output / 'probe_reload.pt', {'state_dict': state})
                    loaded = torch.load(args.output / 'probe_reload.pt', map_location='cpu')['state_dict']
                    assert all(torch.equal(state[k], loaded[k]) for k in state)
                    ema.load_state_dict(loaded, strict=True)
                    with torch.no_grad():
                        output = ema(images[:1])
                        assert set(output[0]) == {'boxes', 'labels', 'scores'}
                        assert all(torch.isfinite(v).all() for v in output[0].values())
                    recheck_data(records, audit)
                    assert sources == inventory(args.source_root)
                    estimate = sum(timings[2:]) / 4 * budget['planned_total_updates'] * 1.15 + 60
                    result = {'status':'PROBE_PASS', 'test_evaluated':False, 'validation_evaluated':False,
                        'protocol_sha256':protocol_sha, 'estimated_full_seconds':estimate,
                        'measured_updates':4, 'timed_seconds':timings[2:],
                        'train_image_reload_inference':'PASS', 'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
                        'method':args.method, 'params':sum(p.numel() for p in model.parameters())}
                    common.write_json(args.output / 'result.json', result)
                    print(json.dumps(result), flush=True)
                    return
        assert samples == audit['expected_counts']['train'] and accumulated == 0
        values = evaluate(ema, val_loader)
        score = .1 * values['AP50'] + .9 * values['AP50_95']
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            common.checkpoint(torch, args.output / 'best.pt', {'state_dict':{k:v.detach().cpu().clone() for k,v in ema.state_dict().items()},
                'epoch':epoch, 'seed':args.seed, 'protocol_sha256':protocol_sha, 'validation':values})
        else:
            stale += 1
        row = {'epoch':epoch, 'optimizer_updates':updates, 'samples':samples, 'train_loss':loss_sum/samples,
               'validation':values, 'best_epoch':best_epoch, 'epoch_seconds':time.perf_counter()-epoch_start}
        common.append_json(args.output / 'epochs.jsonl', row)
        common.write_json(args.output / 'progress.json', row, replace=True)
        print(json.dumps(row), flush=True)
        if stale >= 30:
            break
    checkpoint = torch.load(args.output / 'best.pt', map_location='cpu')
    assert checkpoint['seed'] == args.seed and checkpoint['protocol_sha256'] == protocol_sha
    assert all(not v.is_floating_point() or v.dtype == torch.float32 for v in checkpoint['state_dict'].values())
    ema.load_state_dict(checkpoint['state_dict'], strict=True)
    metrics = evaluate(ema, loader('test'))
    recheck_data(records, audit)
    assert sources == inventory(args.source_root)
    common.write_json(args.output / 'test.json', {'metrics':metrics, 'test_evaluated_once':True,
        'checkpoint_sha256':common.sha(args.output / 'best.pt')})
    # Scalar performance is preserved when a profiler lacks ROI/NMS coverage.
    complexity = {'params':sum(p.numel() for p in ema.parameters()), 'covered_flops':None,
        'complete_operation_coverage':False, 'input':[1,3,640,640], 'precision':'FP32',
        'status':'PARAMETERS_MEASURED_FLOPS_PENDING_ROI_NMS_AUDIT'}
    try:
        from covered_flops import profile_model
        class TensorInput(torch.nn.Module):
            def __init__(self, detector):
                super().__init__()
                self.detector = detector
            def forward(self, tensor):
                return self.detector(list(tensor))
        wrapped = TensorInput(ema).eval()
        sample = torch.zeros((1,3,640,640),device='cuda')
        with torch.no_grad():
            for _ in range(5):
                wrapped(sample)
            torch.cuda.synchronize()
            durations = []
            for _ in range(20):
                begin = time.perf_counter()
                wrapped(sample)
                torch.cuda.synchronize()
                durations.append(time.perf_counter()-begin)
        complexity['latency_ms_mean'] = sum(durations)/len(durations)*1000
        complexity['throughput_samples_per_second'] = 1000/complexity['latency_ms_mean']
        complexity['latency_scope'] = 'batch1 synthetic640 input already on GPU; includes native detector decoding and NMS, excludes file I/O'
        covered = profile_model(wrapped, inputs=(sample,))
        complexity['covered_operation_profile'] = covered
        if covered['status'] == 'PASS':
            complexity['covered_flops'] = covered['covered_flops_g'] * 1e9
            complexity['status'] = 'PARTIAL_COVERAGE_MEASURED'
        complexity['coverage_warning'] = 'ROIAlign, proposals, box decoding, NMS, interpolation and functional normalization/activation arithmetic are not exhaustively counted.'
    except Exception as error:
        complexity['status'] = 'PROFILE_FAILED_PERFORMANCE_RESULTS_PRESERVED'
        complexity['error'] = str(error)
    common.write_json(args.output / 'complexity.json', complexity)
    result = {'status':'TRAINING_COMPLETE', 'seed':args.seed, 'dataset':args.dataset_id, 'method':args.method,
        'metrics':metrics, 'metric_unit':'fraction_0_1', 'protocol_sha256':protocol_sha,
        'checkpoint':{'path':'best.pt','sha256':common.sha(args.output / 'best.pt')},
        'epochs_completed':epoch, 'optimizer_updates':updates, 'best_epoch':best_epoch,
        'elapsed_seconds':time.perf_counter()-start, 'test_evaluated':True,
        'params':complexity['params'], 'complexity':complexity, 'historical_results_reused':False}
    common.write_json(args.output / 'result.json', result)
    common.write_json(args.output / 'completion.json', {'status':'TRAINING_COMPLETE',
        'result_sha256':common.sha(args.output / 'result.json')})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--dataset-manifest', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--dataset-id', choices=['ssdd','sadd','sar_ship'], required=True)
    parser.add_argument('--method', choices=sorted(METHODS), required=True)
    parser.add_argument('--seed', type=int, choices=[0,1,2], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--mode', choices=['probe','train'], required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        if args.output.is_dir():
            common.write_json(args.output / 'failure.json', {'error':str(error), 'type':type(error).__name__, 'traceback':traceback.format_exc()})
        raise
