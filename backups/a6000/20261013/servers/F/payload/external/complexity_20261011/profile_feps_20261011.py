import sys,pathlib,types,torch,json
root=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(root/'mmdetection-2.25.2'))
from mmcv import Config
from mmcv.ops import ModulatedDeformConv2dPack,modulated_deform_conv2d
from mmdet.models import build_detector
from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import get_shape
def deform_conv_ops(inputs,outputs):
    weight=get_shape(inputs[3]);out=get_shape(outputs[0])
    return out[0]*out[1]*out[2]*out[3]*weight[1]*weight[2]*weight[3]
# Compatibility reconstruction of the unpublished extra-offset DCN dependency.
class DCN(ModulatedDeformConv2dPack):
    def __init__(self,*args,deformable_groups=1,extra_offset_mask=False,**kwargs):
        super().__init__(*args,deform_groups=deformable_groups,**kwargs)
    def forward(self,inputs,main_path=None):
        x,feature=inputs
        out=self.conv_offset(feature);o1,o2,mask=torch.chunk(out,3,dim=1)
        offset=torch.cat((o1,o2),dim=1);mask=mask.sigmoid()
        return modulated_deform_conv2d(x,offset,mask,self.weight,self.bias,self.stride,self.padding,self.dilation,self.groups,self.deform_groups)
dcn=types.ModuleType('dcn_v2');dcn.DCN=DCN;sys.modules['dcn_v2']=dcn
name='mmdet.models.necks.feps_staged'
module=types.ModuleType(name);module.__package__='mmdet.models.necks'
exec(compile((root/'FEPS-Net-main/FEPS.py').read_text(),name,'exec'),module.__dict__)
cfg=Config.fromfile(str(root/'mmdetection-2.25.2/configs/retinanet/retinanet_r50_fpn_1x_coco.py'))
cfg.model.backbone.init_cfg=None;cfg.model.backbone.out_indices=(0,1,2,3)
cfg.model.neck=dict(type='FPN_ALL',in_channels=[256,512,1024,2048],out_channels=256)
cfg.model.bbox_head.num_classes=1
cfg.model.bbox_head.anchor_generator.strides=[4,8,16,32,64]
model=build_detector(cfg.model).eval().cuda();torch.set_num_threads(2)
class Wrapper(torch.nn.Module):
    def __init__(self,m):super().__init__();self.model=m
    def forward(self,x):return self.model.forward_dummy(x)
with torch.no_grad():
    profiler=FlopCountAnalysis(Wrapper(model),(torch.zeros(1,3,640,640,device='cuda'),))
    profiler.set_op_handle('prim::PythonOp.ModulatedDeformConv2dFunction',deform_conv_ops)
    ops=profiler.total()
row=dict(method='FEPS-Net-reconstruction',params=sum(p.numel() for p in model.parameters()),flops_g=2*ops/1e9,unsupported=dict(profiler.unsupported_ops()),classes=1,input=[1,3,640,640],status='author neck plus reconstructed detector config and DCN dependency; not confirmed original model')
print(json.dumps(row),flush=True);(root/'feps_results.json').write_text(json.dumps(row,indent=2))
