import importlib.util, json, pathlib
modules = ['torch','thop','mmdet','detectron2','fvcore','ultralytics','mmcv','timm']
print(json.dumps({m: bool(importlib.util.find_spec(m)) for m in modules}))
for p in ['/opt/miniconda3/envs','/home/powerleader/ig-ds-ssm/envs']:
    for env in pathlib.Path(p).glob('*'):
        print('ENV', env)
