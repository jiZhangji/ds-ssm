import sys,pathlib,json,torch,yaml,traceback
root=pathlib.Path(__file__).resolve().parent
source=pathlib.Path('/home/powerleader/ig-ds-ssm/tasks/rgsp_controls_20261008_v3/code')
sys.path.insert(0,str(source))
from ultralytics import YOLO
from thop import profile
torch.set_num_threads(2)
rows=[]
configs=[('YOLOv3',source/'ultralytics/cfg/models/v3/yolov3.yaml'),('YOLOv5n',source/'ultralytics/cfg/models/v5/yolov5.yaml'),('YOLOv5s',source/'ultralytics/cfg/models/v5/yolov5.yaml'),('YOLOv8n',source/'ultralytics/cfg/models/v8/yolov8.yaml')]
manifest=pathlib.Path('/home/powerleader/ig-ds-ssm/tasks/rgsp_recovery_20261008_v1/ssdd_seed0/manifest/configs')
configs += [('Original-clean',manifest/'original.yaml'),('RGSP-clean',manifest/'full.yaml'),('ex27',source/'ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DS-SSM_Backbone-SSDD-ex7.yaml')]
for name,p in configs:
    try:
        cfg=yaml.safe_load(p.read_text());cfg['nc']=1
        if name in ['YOLOv5n','YOLOv5s','YOLOv8n']:cfg['scale']=name[-1]
        target=root/(name+'.yaml');target.write_text(yaml.safe_dump(cfg,sort_keys=False))
        model=YOLO(str(target)).model.eval().cuda()
        with torch.no_grad(): macs,_=profile(model,inputs=(torch.zeros(1,3,640,640,device='cuda'),),verbose=False)
        row=dict(method=name,params=sum(p.numel() for p in model.parameters()),flops_g=2*macs/1e9,config=str(p),classes=1,input=[1,3,640,640],scan_counted=False)
        print(json.dumps(row),flush=True);rows.append(row)
        del model;torch.cuda.empty_cache()
    except Exception:traceback.print_exc()
(root/'local_results.json').write_text(json.dumps(rows,indent=2))
