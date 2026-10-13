"""Build the complete offline gallery from frozen, recorded evidence.

Run this on a suitable inventoried server, not implicitly on the local GPU:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
    python code/build_gallery.py --out .

Dependencies: numpy, matplotlib, Pillow. No torch, CUDA, detector, checkpoint,
network request, training, inference, or synthetic research measurements.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import html
import json
import os
import platform
import textwrap
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ['CUDA_VISIBLE_DEVICES']=''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from common import setup, read_json
import plot_current
import plot_historical
import plot_protocol
import plot_qualitative
import method_diagrams

CATEGORIES={'current':'最新seed0检测消融','multiseed':'独立历史三seed对照','efficiency':'历史成本账本',
            'training':'历史训练曲线','qualitative':'真实样本与历史特征','mechanism':'方法结构示意','protocol':'协议与证据覆盖'}
EVIDENCE={'recorded_single_seed':'真实记录 · 单seed','sadd_multi_seed':'SADD独立三seed',
          'historical_multi_seed':'旧SSDD三seed','historical_ledger':'历史账本 · 非新测速',
          'historical_training':'历史训练 · 非本轮','historical_activations':'历史权重真实特征',
          'historical_diagnostic':'历史诊断 · 非机制证明','real_images':'真实SAR与GT',
          'schematic':'结构/流程示意','protocol_metadata':'协议元数据','snapshot_metadata':'固定时点快照'}
CODE={'C':'plot_current.py','H':'plot_historical.py','Q':'plot_qualitative.py','M':'method_diagrams.py','P':'plot_protocol.py'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def validate_inputs(out):
    manifest=read_json(out/'inputs/source_manifest.json')
    failures=[]
    for row in manifest['entries']:
        path=out/row['staged']
        if not path.is_file() or sha(path)!=row['sha256']:failures.append(row['staged'])
    if failures:raise ValueError('Input hash mismatch: '+str(failures))
    return {'input_files_verified':len(manifest['entries']),'input_hash_failures':failures,
            'snapshot':manifest['source_snapshot']}


def make_sheet(out,rows,name,columns=3,tile_width=700,tile_height=420):
    margin=24;header=66;nrows=(len(rows)+columns-1)//columns
    width=columns*tile_width+(columns+1)*margin;height=nrows*(tile_height+72)+header+margin
    canvas=Image.new('RGB',(width,height),'#F1F4F6');draw=ImageDraw.Draw(canvas)
    font_path=font_manager.findfont('DejaVu Sans')
    large=ImageFont.truetype(font_path,26);small=ImageFont.truetype(font_path,19)
    draw.text((margin,17),'RGSP figure selection | '+name.replace('_',' '),font=large,fill='#283846')
    for index,row in enumerate(rows):
        x=margin+(index%columns)*(tile_width+margin);y=header+(index//columns)*(tile_height+72)
        draw.rectangle((x,y,x+tile_width,y+tile_height+57),fill='white')
        source=Image.open(out/row['files']['png']).convert('RGB')
        source.thumbnail((tile_width-10,tile_height-8),Image.Resampling.LANCZOS)
        canvas.paste(source,(x+(tile_width-source.width)//2,y+(tile_height-source.height)//2))
        text=row['id']+' | '+row['title_en']
        for lineno,line in enumerate(textwrap.wrap(text,63)[:2]):
            draw.text((x+9,y+tile_height+2+lineno*24),line,font=small,fill='#283846')
    target=out/'contact_sheets'/f'{name}.png';canvas.save(target,optimize=True)
    return target.relative_to(out).as_posix()


def make_html(out,rows):
    cards=[]
    for row in rows:
        ident=row['id'];title=html.escape(row['title_cn']);caption=html.escape(row['caption_cn'])
        links=' '.join(f'<a href="{html.escape(path)}" target="_blank" rel="noopener">{fmt.upper()}</a>' for fmt,path in row['files'].items())
        sources=''.join('<li>'+html.escape(s)+'</li>' for s in row['sources'])
        cards.append(f'''<article class="card" data-category="{row['category']}" data-evidence="{row['evidence']}" data-search="{html.escape(ident+' '+row['title_cn']+' '+row['title_en'])}">
<a class="preview" href="{row['files']['png']}" target="_blank" rel="noopener"><img loading="lazy" src="{row['files']['png']}" alt="{title}"></a>
<div class="body"><label class="select"><input type="checkbox" value="{ident}"><strong>{ident}</strong> {title}</label>
<p class="tag">{EVIDENCE.get(row['evidence'],row['evidence'])}</p><nav>{links} <a href="code/{CODE[ident[0]]}">生成代码</a></nav>
<details><summary>图注与来源</summary><p>{caption}</p><ul>{sources}</ul></details></div></article>''')
    options=''.join(f'<option value="{k}">{v}</option>' for k,v in CATEGORIES.items())
    page='''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RGSP 科研选图册</title><style>
:root{font-family:Arial,"Microsoft YaHei",sans-serif;color:#273746;background:#f4f6f8;font-size:15px;line-height:1.6}
*{box-sizing:border-box}body{margin:0}header,main{max-width:1500px;margin:auto;padding:26px}h1{font-size:27px;font-weight:600;margin:0 0 8px}p{margin:8px 0}
.intro{max-width:1050px;color:#586979}.toolbar{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:20px 0}.toolbar label{display:flex;gap:7px;align-items:center}
select,input[type=search],button,textarea{font:inherit;border:1px solid #bfcbd5;border-radius:5px;background:white;color:inherit;padding:8px 10px}
input[type=search]{min-width:220px}button{cursor:pointer}button:hover{background:#eaf1f6}a{color:#286184;text-decoration:none}a:hover{text-decoration:underline}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:20px}.card{background:white;border:1px solid #e0e6eb;border-radius:8px;overflow:hidden}
.preview{display:flex;height:280px;align-items:center;justify-content:center;padding:7px;background:white}.preview img{display:block;max-width:100%;max-height:100%;object-fit:contain}
.body{padding:14px 17px}.select{display:block;cursor:pointer}.select input{margin-right:8px;width:17px;height:17px;vertical-align:middle}.tag{font-size:13px;color:#677789;margin-top:7px}
nav{display:flex;gap:18px;flex-wrap:wrap}details{margin-top:12px;font-size:14px}details p{color:#586979}details ul{padding-left:18px;overflow-wrap:anywhere}
.selection{background:#eaf1f6;padding:16px;border-radius:7px;margin-bottom:22px}.selection textarea{display:block;width:100%;min-height:60px;margin-top:8px;font-family:monospace}
.warning{border-left:3px solid #c09155;padding-left:12px;color:#64523a}.hidden{display:none}footer{padding:24px;color:#657687;font-size:13px}
@media(max-width:600px){header,main{padding:16px}.preview{height:240px}.grid{grid-template-columns:1fr}}
</style></head><body><header><h1>RGSP 科研选图册</h1>
<p class="intro">50 张候选图，每张保留 PNG、可编辑 SVG、矢量 PDF 和生成代码。先按类别浏览，勾选后把图号发回即可继续精修。所有数据固定于本包来源快照，不自动连接服务器。</p>
<p class="warning">最新消融只有 seed0；历史三seed与当前协议分开。真实热图来自历史 ex27 权重，并非当前 P3 的新推理。示意图不作为实验或机制证据。缺失结果未填0。</p>
<p><a href="gallery_booklet.pdf">整本 PDF</a> · <a href="contact_sheets/START_HERE.png">推荐图总览</a> · <a href="README.md">复现与选图说明</a> · <a href="manifest.json">完整图片索引</a> · <a href="notes/conversation_audit.md">对话覆盖记录</a></p></header>
<main><div class="toolbar"><label>类别 <select id="category"><option value="all">全部</option>'''+options+'''</select></label><label>搜索 <input id="search" type="search" placeholder="图号、数据集或关键词"></label>
<label><input id="selected-only" type="checkbox">只看已选</label><span id="count"></span></div>
<section class="selection"><label for="selected">已选图号</label><textarea id="selected" readonly placeholder="勾选图片后，图号会显示在这里"></textarea><button id="export" type="button">下载选图清单</button> <button id="clear" type="button">清空已选</button></section>
<div class="grid">'''+''.join(cards)+'''</div></main><footer>图内标签为英文，图注与说明为中文。离线目录，无外部字体、分析脚本或网络请求；选择只保存在本浏览器。</footer>
<script>
const storageKey='rgsp-figure-choice-20261010';let selected=new Set();try{selected=new Set(JSON.parse(localStorage.getItem(storageKey)||'[]'))}catch{}
const cards=[...document.querySelectorAll('.card')],boxes=[...document.querySelectorAll('.select input')];
function update(){const cat=document.getElementById('category').value,q=document.getElementById('search').value.trim().toLowerCase(),only=document.getElementById('selected-only').checked;let n=0;
cards.forEach(card=>{const id=card.querySelector('.select input').value,show=(cat==='all'||cat===card.dataset.category)&&card.dataset.search.toLowerCase().includes(q)&&(!only||selected.has(id));card.classList.toggle('hidden',!show);n+=show?1:0});
document.getElementById('count').textContent=n+' 张可见 / '+selected.size+' 张已选';document.getElementById('selected').value=[...selected].sort().join(', ')}
boxes.forEach(box=>{box.checked=selected.has(box.value);box.addEventListener('change',()=>{box.checked?selected.add(box.value):selected.delete(box.value);try{localStorage.setItem(storageKey,JSON.stringify([...selected]))}catch{}update()})});
['category','search','selected-only'].forEach(id=>document.getElementById(id).addEventListener('input',update));
document.getElementById('clear').addEventListener('click',()=>{selected.clear();boxes.forEach(b=>b.checked=false);try{localStorage.removeItem(storageKey)}catch{}update()});
document.getElementById('export').addEventListener('click',()=>{const blob=new Blob(['RGSP selected figures\\n'+[...selected].sort().join(', ')+'\\n'],{type:'text/plain;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='rgsp-selected-figures.txt';a.click();URL.revokeObjectURL(url)});update();
</script></body></html>'''
    (out/'index.html').write_text(page,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',default=str(Path(__file__).resolve().parents[1]));parser.add_argument('--dpi',type=int,default=300)
    args=parser.parse_args();out=Path(args.out).resolve();setup()
    check=validate_inputs(out)
    for name in ['png','svg','pdf','contact_sheets','derived']: (out/name).mkdir(exist_ok=True)
    rows=[];layout=[]
    meta={'Title':'RGSP research figure collection','Author':'RGSP research workspace',
          'Subject':'Recorded research evidence and explicitly labeled schematics',
          'CreationDate':datetime.datetime(2026,10,10,tzinfo=datetime.timezone.utc),
          'ModDate':datetime.datetime(2026,10,10,tzinfo=datetime.timezone.utc)}
    with PdfPages(out/'gallery_booklet.pdf',metadata=meta) as book:
        def save(fig,ident,title_cn,category,caption_cn,sources,evidence='schematic'):
            if any(r['id']==ident for r in rows):raise ValueError('Duplicate figure ID: '+ident)
            if category not in CATEGORIES:raise ValueError('Unknown category: '+category)
            title_en=fig._suptitle.get_text() if fig._suptitle else (fig.texts[0].get_text() if fig.texts else ident)
            fig.text(.985,.993,ident,ha='right',va='top',fontsize=9,color='#687787')
            fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox
            outside=[]
            for text in fig.findobj(match=matplotlib.text.Text):
                if not text.get_visible() or not text.get_text():continue
                if hasattr(text,'get_clip_on') and text.get_clip_on():continue
                extent=text.get_window_extent(renderer)
                if extent.width==0 or extent.height==0:continue
                if extent.x0<bounds.x0-2 or extent.x1>bounds.x1+2 or extent.y0<bounds.y0-2 or extent.y1>bounds.y1+2:
                    outside.append(text.get_text()[:120])
            layout.append({'id':ident,'outside_nominal_canvas_text':outside})
            files={fmt:f'{fmt}/{ident}.{fmt}' for fmt in ['png','svg','pdf']}
            fig.savefig(out/files['png'],dpi=args.dpi,bbox_inches='tight',pad_inches=.08)
            fig.savefig(out/files['svg'],format='svg',bbox_inches='tight',pad_inches=.08,
                        metadata={'Title':title_en,'Description':caption_cn,'Date':'2026-10-10'})
            fig.savefig(out/files['pdf'],format='pdf',bbox_inches='tight',pad_inches=.08,
                        metadata={**meta,'Title':title_en})
            book.savefig(fig,bbox_inches='tight',pad_inches=.08)
            rows.append({'id':ident,'title_cn':title_cn,'title_en':title_en,'category':category,'evidence':evidence,
                         'caption_cn':caption_cn,'sources':sources,'code':'code/'+CODE[ident[0]],'files':files,
                         'sha256':{fmt:sha(out/path) for fmt,path in files.items()}})
            plt.close(fig)
            print(json.dumps({'saved':ident,'title':title_en},ensure_ascii=True),flush=True)
        for module in [plot_current,plot_historical,plot_qualitative,method_diagrams,plot_protocol]:
            module.build(save,out/'inputs')
    expected=set([f'C{i:02d}' for i in range(1,17)]+[f'H{i:02d}' for i in range(1,13)]+
                 [f'Q{i:02d}' for i in range(1,13)]+[f'M{i:02d}' for i in range(1,7)]+[f'P{i:02d}' for i in range(1,5)])
    if {r['id'] for r in rows}!=expected:raise ValueError('The expected 50 figures were not all produced')
    sheets=[]
    chosen=['C01','C05','C10','C11','C13','C14','H01','H05','Q07','Q08','M01','M05']
    sheets.append(make_sheet(out,[next(r for r in rows if r['id']==i) for i in chosen],'START_HERE'))
    for category in CATEGORIES:
        subset=[r for r in rows if r['category']==category]
        sheets.append(make_sheet(out,subset,category,columns=3 if len(subset)>3 else 2,tile_width=650,tile_height=410))
    make_html(out,rows)
    result={'schema':'rgsp-figure-gallery-v1','snapshot':check['snapshot'],'figure_count':len(rows),
            'image_format':'300-dpi PNG; editable-text SVG; embedded-font PDF','categories':CATEGORIES,
            'evidence_labels':EVIDENCE,'figures':rows,'contact_sheets':sheets,
            'booklet':{'file':'gallery_booklet.pdf','sha256':sha(out/'gallery_booklet.pdf')},
            'research_runs_started':0,'new_inference_runs':0}
    (out/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    check.update({'figure_count':len(rows),'format_files':len(rows)*3,'ids_unique':True,
                  'expected_ids_match':True,'layout_nominal_canvas_scan':layout,
                  'python':platform.python_version(),'matplotlib':matplotlib.__version__,'numpy':np.__version__,
                  'host':platform.node(),'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
                  'research_validation':False,'note':'Artifact/data-integrity checks only, not new model/experiment validation.'})
    (out/'derived/build_checks.json').write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'complete':True,'figures':len(rows),'formats':150,'input_files':check['input_files_verified']}),flush=True)


if __name__=='__main__':main()
