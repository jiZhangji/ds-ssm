import pathlib,urllib.request,zipfile,json,shutil
root=pathlib.Path(__file__).resolve().parent
url='https://codeload.github.com/open-mmlab/mmdetection/zip/refs/tags/v2.25.2'
archive=root/'mmdetection_2.25.2.zip';urllib.request.urlretrieve(url,archive)
with zipfile.ZipFile(archive) as z:z.extractall(root)
changes=[]
for source in (root/'mmdetection-2.25.2/mmdet').rglob('*.py'):
    rel=source.relative_to(root/'mmdetection-2.25.2')
    target=root/'FBGBNet-master'/rel
    if not target.exists():
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);changes.append(str(rel))
(root/'fgbg_missing_framework_files.json').write_text(json.dumps(changes,indent=2))
print('Restored missing upstream framework files:',len(changes),changes,flush=True)
