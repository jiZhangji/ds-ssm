"""Low-load artifact checks and independent Poppler PDF rendering.

Run on the selected server after build_gallery.py. This validates files and
figure exports, not the original research experiment or detector behavior.
"""
import argparse
import ast
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image, ImageStat


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',default=str(Path(__file__).resolve().parents[1]));args=parser.parse_args()
    out=Path(args.out).resolve();qa=out/'qa';renders=qa/'pdf_renders';renders.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'));failures=[];rows=[]
    for path in (out/'code').glob('*.py'):ast.parse(path.read_text(encoding='utf-8'),filename=path.name)
    for item in manifest['figures']:
        ident=item['id'];row={'id':ident}
        for fmt,relative in item['files'].items():
            path=out/relative
            if not path.is_file() or path.stat().st_size<1000 or sha(path)!=item['sha256'][fmt]:failures.append(f'{ident}: {fmt} absent or hash mismatch')
        with Image.open(out/item['files']['png']) as im:
            row['png_size']=list(im.size)
            if min(im.size)<700:failures.append(ident+': PNG too small')
            if max(ImageStat.Stat(im.convert('RGB')).stddev)<4:failures.append(ident+': PNG appears blank')
        svg=ET.parse(out/item['files']['svg']);texts=svg.findall('.//{http://www.w3.org/2000/svg}text')
        row['svg_editable_text_nodes']=len(texts)
        if len(texts)<4:failures.append(ident+': SVG text not editable')
        info=subprocess.run(['pdfinfo',str(out/item['files']['pdf'])],capture_output=True,text=True,check=True).stdout
        match=re.search(r'^Pages:\s+(\d+)',info,re.M)
        row['pdf_pages']=int(match.group(1)) if match else None
        if row['pdf_pages']!=1:failures.append(ident+': PDF page count')
        fonts=subprocess.run(['pdffonts',str(out/item['files']['pdf'])],capture_output=True,text=True,check=True).stdout
        row['fonts_report']=fonts
        if 'Type 3' in fonts:failures.append(ident+': Type 3 font')
        for line in fonts.splitlines()[2:]:
            # Columns from the right: embedded, subset, unicode, object, ID.
            cols=line.split()
            if len(cols)>=5 and cols[-5]!='yes':failures.append(ident+': unembedded PDF font')
        subprocess.run(['pdftoppm','-f','1','-singlefile','-r','110','-png',str(out/item['files']['pdf']),str(renders/ident)],check=True,capture_output=True)
        row['pdf_render_sha256']=sha(renders/f'{ident}.png');rows.append(row)
    info=subprocess.run(['pdfinfo',str(out/'gallery_booklet.pdf')],capture_output=True,text=True,check=True).stdout
    pages=int(re.search(r'^Pages:\s+(\d+)',info,re.M).group(1))
    if pages!=50:failures.append('Booklet is not 50 pages')
    # Validate every local HTML href/src (fragments and external references excluded).
    html=(out/'index.html').read_text(encoding='utf-8')
    missing_links=[]
    for ref in re.findall(r'(?:href|src)="([^"]+)"',html):
        if ref.startswith(('http:','https:','#','data:')):continue
        if not (out/ref).exists():missing_links.append(ref)
    failures.extend('Missing gallery link: '+x for x in sorted(set(missing_links)))
    result={'schema':'rgsp-figure-artifact-validation-v1','status':'STRUCTURAL_PASS' if not failures else 'FAILED',
            'figure_count':len(rows),'pdf_booklet_pages':pages,'all_pdf_pages_rendered':True,
            'failures':failures,'figures':rows,
            'visual_review':{'status':'not_performed_by_this_script','report':'notes/EXECUTION_AND_QA.md'},
            'research_validation':False,'new_inference_runs':0}
    (qa/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    # A contact sheet uses the independent PDF render, not the PNG export.
    import build_gallery
    for category in build_gallery.CATEGORIES:
        selection=[]
        for original in manifest['figures']:
            if original['category']==category:
                item=dict(original);item['files']={'png':f'qa/pdf_renders/{item["id"]}.png'};selection.append(item)
        build_gallery.make_sheet(out,selection,'PDF_QA_'+category,columns=3 if len(selection)>3 else 2,tile_width=680,tile_height=435)
    print(json.dumps({'status':result['status'],'figures':len(rows),'pages':pages,'failures':failures}),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
