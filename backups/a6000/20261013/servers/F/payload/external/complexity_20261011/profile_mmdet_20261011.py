import pathlib,json,torch, traceback
from mmengine.config import Config
from mmengine.analysis import FlopAnalyzer
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules
from mmdet.structures import DetDataSample
register_all_modules()
torch.set_num_threads(2)
root=pathlib.Path(__file__).resolve().parent
import mmdet
configs=pathlib.Path(mmdet.__file__).parent/'.mim/configs'
class Wrapper(torch.nn.Module):
    def __init__(self,m): super().__init__(); self.model=m
    def forward(self,x):
        sample=DetDataSample();sample.set_metainfo(dict(img_shape=(640,640),ori_shape=(640,640),pad_shape=(640,640),scale_factor=(1.,1.)))
        return self.model._forward(x,[sample])
rows=[]
for name,config in [('FCOS-R50-FPN','fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py'),('Cascade-RCNN-R50-FPN','cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py')]:
    try:
        cfg=Config.fromfile(str(configs/config)); cfg.model.backbone.init_cfg=None
        head=cfg.model.get('bbox_head',cfg.model.get('roi_head',{}).get('bbox_head'))
        for h in head if isinstance(head,list) else [head]: h.num_classes=1
        model=MODELS.build(cfg.model).eval().cuda()
        x=torch.zeros(1,3,640,640,device='cuda')
        with torch.no_grad():
            analyzer=FlopAnalyzer(Wrapper(model),(x,));ops=analyzer.total()
        row=dict(method=name,params=sum(p.numel() for p in model.parameters()),flops_g=2*ops/1e9,unsupported=dict(analyzer.unsupported_ops()),config=config,classes=1,input=[1,3,640,640])
        print(json.dumps(row),flush=True); rows.append(row)
        del model;torch.cuda.empty_cache()
    except Exception: traceback.print_exc()
(root/'mmdet_results.json').write_text(json.dumps(rows,indent=2))
