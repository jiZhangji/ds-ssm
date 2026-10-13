import pathlib,urllib.request,zipfile,json,hashlib
root=pathlib.Path(__file__).resolve().parent
for repo in ['JoyeZLearning/MaDiNet','so-bright/FEPS-Net','Duankaiwen/CenterNet']:
    for branch in ['main','master']:
        try:
            url=f'https://codeload.github.com/{repo}/zip/refs/heads/{branch}'
            target=root/(repo.split('/')[-1]+'_small.zip')
            urllib.request.urlretrieve(url,target)
            with zipfile.ZipFile(target) as z: z.extractall(root)
            print(json.dumps(dict(repo=repo,branch=branch,sha256=hashlib.sha256(target.read_bytes()).hexdigest())),flush=True)
            break
        except Exception as e: print(repo,branch,repr(e),flush=True)
