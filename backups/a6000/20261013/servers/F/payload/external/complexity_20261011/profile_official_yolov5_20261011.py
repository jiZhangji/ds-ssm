import pathlib,sys,urllib.request,zipfile,torch,json
root=pathlib.Path(__file__).resolve().parent
archive=root/'yolov5_v7.zip'
urllib.request.urlretrieve('https://codeload.github.com/ultralytics/yolov5/zip/refs/tags/v7.0',archive)
with zipfile.ZipFile(archive) as z:z.extractall(root)
sys.path.insert(0,str(root/'yolov5-7.0'))
from models.yolo import Model
from thop import profile
torch.set_num_threads(2);rows=[]
for scale in ['n','s','v3']:
    path=root/f'yolov5-7.0/models/yolov5{scale}.yaml' if scale!='v3' else root/'yolov5-7.0/models/hub/yolov3.yaml'
    model=Model(str(path),ch=3,nc=1).eval().cuda()
    with torch.no_grad():macs,_=profile(model,inputs=(torch.zeros(1,3,640,640,device='cuda'),),verbose=False)
    row=dict(method='YOLOv5'+scale if scale!='v3' else 'YOLOv3',params=sum(p.numel() for p in model.parameters()),flops_g=2*macs/1e9,classes=1,input=[1,3,640,640],source='ultralytics/yolov5 v7.0')
    rows.append(row);print(json.dumps(row),flush=True)
    del model;torch.cuda.empty_cache()
(root/'official_yolov5_results.json').write_text(json.dumps(rows,indent=2))
