import pathlib, urllib.request, zipfile, json, hashlib
root=pathlib.Path(__file__).resolve().parent
for repo, branch in [('Megvii-BaseDetection/YOLOX','main'),('Xidian-AIGroup190726/FBGBNet','master'),('JoyeZLearning/MaDiNet','master'),('so-bright/FEPS-Net','main')]:
    try:
        url=f'https://codeload.github.com/{repo}/zip/refs/heads/{branch}'
        target=root/(repo.split('/')[-1]+'.zip')
        urllib.request.urlretrieve(url,target)
        with zipfile.ZipFile(target) as z: z.extractall(root)
        print(json.dumps({'repo':repo,'branch':branch,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}),flush=True)
    except Exception as e: print(repo,repr(e),flush=True)
