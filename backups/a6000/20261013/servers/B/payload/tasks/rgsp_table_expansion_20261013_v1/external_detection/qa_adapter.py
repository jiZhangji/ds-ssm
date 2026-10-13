"""Server CPU fixtures for labels, IoU matching and official constructors.

No training, forward inference or GPU access. GPU preflight remains separate.
"""
import ast
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / 'external_detection/qa_settings')
os.environ['MPLCONFIGDIR'] = str(ROOT / 'external_detection/qa_matplotlib')
sys.path.insert(0, str(ROOT / 'external_detection'))
sys.path.insert(0, '/home/powerleader/ig-ds-ssm/tasks/rgsp_seedwise_20261009_v1/code')
from run_torchvision import batch_to_targets, make_model, METHODS
import torch
from ultralytics.models.yolo.detect import DetectionValidator

ast.parse((ROOT / 'external_detection/run_torchvision.py').read_text())
batch = {'img': torch.zeros((2, 3, 32, 64), dtype=torch.uint8),
         'batch_idx': torch.tensor([0]), 'cls': torch.tensor([[0.]]),
         'bboxes': torch.tensor([[.5, .5, .5, .5]])}
images, targets = batch_to_targets(batch, 'cpu')
assert torch.equal(targets[0]['boxes'], torch.tensor([[16.,8.,48.,24.]]))
assert targets[0]['labels'].tolist() == [1]
assert targets[1]['boxes'].shape == (0,4) and targets[1]['labels'].shape == (0,)
validator = DetectionValidator(args={'plots': False})
validator.iouv = torch.linspace(.5,.95,10)
pred = torch.tensor([[0.,0.,20.,20.,.9,0.], [0.,0.,20.,20.,.8,0.]])
correct = validator._process_batch(pred, torch.tensor([[0.,0.,20.,20.]]), torch.tensor([0.]))
assert correct.sum().item() == 10 and correct.shape == (2,10)
assert not validator._process_batch(pred, torch.empty((0,4)), torch.empty(0)).any()
assert validator._process_batch(torch.empty((0,6)), torch.tensor([[0.,0.,20.,20.]]), torch.tensor([0.])).shape == (0,10)
constructors = []
for method in sorted(METHODS):
    model, metadata = make_model(method,1)
    assert next(model.parameters()).device.type == 'cpu'
    assert model.transform.min_size == (640,) and model.transform.max_size == 640
    constructors.append({'method':method,'params':sum(p.numel() for p in model.parameters()),'metadata':metadata})
    del model
result = {'status':'PASS', 'syntax':True, 'target_conversion':True, 'empty_targets':True,
          'unique_iou_matching':True, 'empty_predictions':True, 'official_constructors':constructors,
          'gpu_preflight':'PENDING', 'model_forward_or_training':False}
with (ROOT / 'external_detection/qa_adapter_result.json').open('x') as handle:
    json.dump(result,handle,indent=2)
print(json.dumps(result))
