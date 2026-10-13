import sys, pathlib, json, torch
root=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(root/'YOLOX-main'))
from yolox.exp import get_exp
from thop import profile
torch.set_num_threads(2)
rows=[]
for name in ['yolox-nano','yolox-tiny','yolox-s','yolox-m','yolox-l','yolox-x']:
    try:
        exp=get_exp(None,name); exp.num_classes=1
        model=exp.get_model().eval().cuda()
        with torch.no_grad(): macs,params=profile(model,inputs=(torch.zeros(1,3,640,640,device='cuda'),),verbose=False)
        row={'method':name,'params':sum(p.numel() for p in model.parameters()),'flops_g':2*macs/1e9,'input':[1,3,640,640],'classes':1}
        rows.append(row); print(json.dumps(row),flush=True)
        del model; torch.cuda.empty_cache()
    except Exception as e: print(name,repr(e),flush=True)
(root/'yolox_results.json').write_text(json.dumps(rows,indent=2))
