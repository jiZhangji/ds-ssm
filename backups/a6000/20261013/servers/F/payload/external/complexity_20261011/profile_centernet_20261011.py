import sys,pathlib,json,types,torch,importlib
from thop import profile
root=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(root/'CenterNet-master'))
# Corner pooling is parameter-free. Use the same directional cumulative max
# operation in PyTorch to avoid the original Python 3.6-only extension.
poolmodule=types.ModuleType('models.py_utils._cpools')
for name,dim,reverse in [('TopPool',2,True),('BottomPool',2,False),('LeftPool',3,True),('RightPool',3,False)]:
    class Pool(torch.nn.Module):
        def forward(self,x,dim=dim,reverse=reverse):
            if reverse: return torch.flip(torch.cummax(torch.flip(x,[dim]),dim)[0],[dim])
            return torch.cummax(x,dim)[0]
    setattr(poolmodule,name,Pool)
sys.modules['models.py_utils._cpools']=poolmodule
torch.set_num_threads(2)
class DB:
    configs=dict(top_k=100,ae_threshold=1,nms_kernel=3,input_size=[640,640],output_sizes=[[160,160]])
rows=[]
for variant in ['52','104']:
    path=root/'CenterNet-master/models'/('CenterNet-'+variant+'.py')
    module=types.ModuleType('models.profile'+variant);module.__package__='models'
    exec(compile(path.read_text().replace('out_dim = 80','out_dim = 1'),str(path),'exec'),module.__dict__)
    model=module.model(DB()).eval().cuda();model._decode=lambda *outs,**kwargs: outs
    with torch.no_grad():macs,_=profile(model,inputs=(torch.zeros(1,3,640,640,device='cuda'),),verbose=False)
    row=dict(method='CenterNet-HG'+variant,params=sum(p.numel() for p in model.parameters()),flops_g=2*macs/1e9,classes=1,input=[1,3,640,640],pooling_counted=False,decode_counted=False)
    print(json.dumps(row),flush=True);rows.append(row)
    del model;torch.cuda.empty_cache()
(root/'centernet_results.json').write_text(json.dumps(rows,indent=2))
