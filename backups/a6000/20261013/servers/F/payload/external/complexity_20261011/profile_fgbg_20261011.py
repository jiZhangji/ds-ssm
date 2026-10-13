import sys,pathlib,torch,json
root=pathlib.Path(__file__).resolve().parent
# The author file imports two unused image metrics. Remove only these imports
# in the task's staged copy; no layers or forward computations are changed.
head=root/'FBGBNet-master/mmdet/models/dense_heads/vfnet_head.py'
text=head.read_text()
text=text.replace('from skimage.metrics import structural_similarity as compare_ssim','')
text=text.replace('from torchmetrics import StructuralSimilarityIndexMeasure','')
head.write_text(text)
sys.path.insert(0,str(root/'FBGBNet-master'))
from mmcv import Config
from mmdet.models import build_detector
from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import get_shape
def deform_conv_ops(inputs,outputs):
    weight=get_shape(inputs[2]);out=get_shape(outputs[0])
    return out[0]*out[1]*out[2]*out[3]*weight[1]*weight[2]*weight[3]
torch.set_num_threads(2)
cfg=Config.fromfile(str(root/'FBGBNet-master/configs/vfnet/vfnet_r50_fpn_1x_coco.py'))
cfg.model.backbone.init_cfg=None
model=build_detector(cfg.model).eval().cuda()
class Wrapper(torch.nn.Module):
    def __init__(self,m):super().__init__();self.model=m
    def forward(self,x):return self.model.forward_dummy(x)
with torch.no_grad():
    profiler=FlopCountAnalysis(Wrapper(model),(torch.zeros(1,3,640,640,device='cuda'),))
    profiler.set_op_handle('prim::PythonOp.DeformConv2dFunction',deform_conv_ops)
    ops=profiler.total()
row=dict(method='FGBG-Net-author-R50',params=sum(p.numel() for p in model.parameters()),flops_g=2*ops/1e9,unsupported=dict(profiler.unsupported_ops()),classes=cfg.model.bbox_head.num_classes,input=[1,3,640,640],config='vfnet_r50_fpn_1x_coco.py',training_only_branches_excluded=True)
print(json.dumps(row),flush=True)
(root/'fgbg_results.json').write_text(json.dumps(row,indent=2))
