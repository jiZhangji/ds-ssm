"""Replot saved exact-v2 activations. No model import or inference is allowed.

The 2026-08-23 capture used a historical ex27 checkpoint and block_index=0.
It is not an activation audit of the current P3 model. Original photographs
are SHA-matched official TEST images; the older 'val' path is not a new split.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from common import C, read_json, read_csv, finish, grid, COLORS

IDS=['000099','000531','000929','001081','001131','001111']
MAPS=['input_energy','residual','corrected_energy','importance','drive_gate','delta_bias','correction']
TITLES={'input_energy':'Input feature RMS','corrected_energy':'Corrected feature RMS',
        'residual':'Local residual RMS','correction':'Applied correction RMS',
        'importance':'Recorded importance I','drive_gate':'Effective scan-drive gate','delta_bias':'Delta bias'}
NOTE='Historical ex27 | saved activations from 2026-08-23, block 0 | not the current P3 model'


def crop_content(raw,image_shape):
    """Original capture's feature-space letterbox crop, without cubic resize."""
    ih,iw=image_shape;fh,fw=raw.shape
    scale=min(fw/iw,fh/ih)
    cw=max(1,min(fw,int(round(iw*scale))));ch=max(1,min(fh,int(round(ih*scale))))
    left=max(0,(fw-cw)//2);top=max(0,(fh-ch)//2)
    return raw[top:top+ch,left:left+cw]


def build(save,data_dir):
    root=Path(data_dir);coco=read_json(root/'ssdd_test_coco.json')
    image_meta={Path(r['file_name']).stem:r for r in coco['images']}
    imgs={};arrays={};boxes={}
    for sid in IDS:
        imgs[sid]=np.asarray(Image.open(root/'images'/f'{sid}.jpg').convert('L'))
        rec=image_meta[sid]
        if tuple(imgs[sid].shape)!=(rec['height'],rec['width']):raise ValueError('Image/annotation dimensions differ')
        boxes[sid]=[a['bbox'] for a in coco['annotations'] if a['image_id']==rec['id']]
        arrays[sid]={}
        for key in MAPS:
            raw=np.load(root/'arrays'/sid/f'raw_{key}.npy',allow_pickle=False)
            if raw.ndim!=2 or not np.isfinite(raw).all():raise ValueError('Invalid recorded feature array')
            arrays[sid][key]=crop_content(raw,imgs[sid].shape)
    bounds={}
    for key in MAPS:
        lo=min(float(arrays[s][key].min()) for s in IDS)
        hi=max(float(arrays[s][key].max()) for s in IDS)
        bounds[key]=(min(0,lo),hi) if key in {'importance','residual','correction','drive_gate'} else (lo,hi)
    energy=(min(bounds[k][0] for k in ['input_energy','corrected_energy']),
            max(bounds[k][1] for k in ['input_energy','corrected_energy']))
    bounds['input_energy']=bounds['corrected_energy']=energy
    db=max(abs(x) for x in bounds['delta_bias']);bounds['delta_bias']=(-db,db)
    common_sources=['inputs/historical_maps_manifest.json','inputs/historical_capture_source.py','inputs/ssdd_test_coco.json']
    boundary=('复用2026-08-23 exact_v2历史权重的block0已保存特征，不是当前P3模型推理。'
              '6图与历史manifest SHA相同，实际对应官方test；全部固定样本保留。'
              '原生网格按原脚本裁letterbox，不做双三次插值；跨图每种量使用同一色标，输入/修正能量共用色标。'
              'GT是从官方COCO框重新绘制；旧捕获未哈希标签，框并集不是像素分割真值。'
              '热图不是可靠性概率、因果证明或噪声标签，原始灰度不用于识别噪声。')
    archived_summary_boundary=('复用2026-08-23 exact_v2历史权重block0的已存aligned-map统计，不是当前P3模型推理。'
                               '6图与历史manifest SHA相同，实际对应官方test；全部固定样本保留。'
                               '历史aligned映射先裁letterbox再双三次插值，本轮直接引用保存的统计而不重算。'
                               '分区来自原捕获使用的GT框并集，不是本次官方COCO框重算；旧捕获未哈希标签，框并集不是像素分割真值。'
                               '这些特征统计不是可靠性概率、因果证明或噪声标签。')

    def gt(ax,sid,with_title=True):
        ax.imshow(imgs[sid],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
        for x,y,w,h in boxes[sid]:ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='#E6B65B',linewidth=1))
        if with_title:ax.set_title(f'SAR + GT boxes (n={len(boxes[sid])})',fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)

    for idx,sid in enumerate(IDS,1):
        fig,axes=plt.subplots(2,4,figsize=(12,6.8))
        gt(axes.flat[0],sid)
        ih,iw=imgs[sid].shape
        for ax,key in zip(list(axes.flat)[1:],MAPS):
            lo,hi=bounds[key]
            cm='RdBu_r' if key=='delta_bias' else 'viridis' if key in ['importance','drive_gate'] else 'cividis'
            im=ax.imshow(arrays[sid][key],cmap=cm,vmin=lo,vmax=hi,extent=[0,iw,ih,0],interpolation='nearest')
            ax.set_title(TITLES[key],fontsize=10);ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(False)
            fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.065,pad=.07,format='%.2f')
        finish(fig,f'Saved feature maps: SSDD {sid}',NOTE+'\nValid-content native grids; one shared scale per quantity across all six samples. Feature values are not physical SAR intensity.',
               bottom=.15,top=.88)
        fig.subplots_adjust(left=.025,right=.975,wspace=.27,hspace=.42)
        save(fig,f'Q{idx:02d}',f'真实历史特征八面板：样本{sid}','qualitative',boundary,
             common_sources+[f'inputs/images/{sid}.jpg']+[f'inputs/arrays/{sid}/raw_{k}.npy' for k in MAPS],
             evidence='historical_activations')

    fig,axes=plt.subplots(2,3,figsize=(10.8,7.6))
    for ax,sid in zip(axes.flat,IDS):
        gt(ax,sid,False);ih,iw=imgs[sid].shape
        im=ax.imshow(arrays[sid]['importance'],cmap='viridis',vmin=bounds['importance'][0],vmax=bounds['importance'][1],
                     alpha=.56,extent=[0,iw,ih,0],interpolation='nearest')
        ax.set_title(f'{sid} | {len(boxes[sid])} GT boxes',fontsize=10)
    finish(fig,'Historical importance overlays: every saved sample',NOTE+'\nUniform grayscale 0-255; uniform importance scale; overlay alpha = 0.56. Not a target mask.',bottom=.17,top=.89)
    fig.subplots_adjust(left=.03,right=.9,wspace=.18,hspace=.28)
    cax=fig.add_axes([.925,.27,.017,.48]);fig.colorbar(im,cax=cax,label='Recorded I (feature-space scalar)')
    save(fig,'Q07','六张样本的统一色标重要性叠图','qualitative',boundary+'叠加仅用于空间观察，灰度统一0–255且alpha固定0.56。',
         common_sources+[f'inputs/arrays/{s}/raw_importance.npy' for s in IDS]+[f'inputs/images/{s}.jpg' for s in IDS],
         evidence='historical_activations')

    summary=read_csv(root/'historical_maps_summary.csv')
    stat={Path(r['image']).stem:r for r in summary}
    fig,ax=plt.subplots(figsize=(9.5,5.5));y=np.arange(6)
    for i,sid in enumerate(IDS):
        a=float(stat[sid]['importance_fg_mean']);b=float(stat[sid]['importance_bg_mean'])
        ax.plot([a,b],[i,i],color='#A9B5BF',linewidth=2)
        ax.scatter(a,i,color=C['orange'],marker='o',s=65,label='GT-box foreground' if i==0 else None)
        ax.scatter(b,i,color=C['blue'],marker='s',s=60,label='Outside GT boxes' if i==0 else None)
        ax.text(.405,i,f'FG - BG = {a-b:+.3f}',va='center',fontsize=9)
    ax.set_yticks(y,IDS);ax.invert_yaxis();ax.set_xlim(-.005,.61)
    ax.set_xlabel('Archived aligned-map mean importance (feature-space scalar)');ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),ncol=2,fontsize=9);grid(ax)
    finish(fig,'Importance is not consistently higher inside target boxes',NOTE+'\nExact archived summary values; foreground = box union, not a segmentation mask. No population-level inference.',bottom=.2,top=.79)
    fig.subplots_adjust(left=.11,right=.98)
    save(fig,'Q08','前景与背景重要性：保留反例','qualitative',
         '直接读取原aligned-map统计CSV，不将像素总体std当成实验误差条。4/6样本前景均值低于背景是这6图的描述，不外推总体。与新绘native-grid图存在采样/插值口径差异。'+archived_summary_boundary,
         ['inputs/historical_maps_summary.csv','inputs/historical_maps_manifest.json'],evidence='historical_diagnostic')

    fig,axes=plt.subplots(1,2,figsize=(11,5.2))
    for ax,region,title in zip(axes,['fg','bg'],['Inside GT boxes','Outside GT boxes']):
        for i,sid in enumerate(IDS):
            a=float(stat[sid][f'input_energy_{region}_mean']);b=float(stat[sid][f'corrected_energy_{region}_mean'])
            ax.plot([a,b],[i,i],color='#ACB6BF',linewidth=1.7)
            ax.scatter(a,i,color=C['blue'],s=42,label='Input' if i==0 else None)
            ax.scatter(b,i,color=C['teal'],s=42,marker='s',label='Corrected' if i==0 else None)
        ax.set_yticks(range(6),IDS);ax.invert_yaxis();ax.set_xlim(.78,1.12);ax.set_xlabel('Mean feature RMS');ax.set_title(title);grid(ax)
    axes[0].legend(loc='upper left')
    finish(fig,'Energy changes within the historical feature pipeline',NOTE+'\nArchived aligned-map summary. A decrease in feature RMS is not proof of denoising or target preservation.',bottom=.21,top=.83)
    fig.subplots_adjust(left=.095,wspace=.25)
    save(fig,'Q09','前景和背景修正前后的能量对照','qualitative',
         '直接比较同一历史模块输入与修正后的特征RMS；不把能量降低等同于去噪成功，更不把特征RMS称为物理SAR强度。'+archived_summary_boundary,
         ['inputs/historical_maps_summary.csv','inputs/historical_maps_manifest.json'],evidence='historical_diagnostic')

    fig,ax=plt.subplots(figsize=(9.4,5.1))
    for sid,col,style in zip(IDS,COLORS,['-','--','-.',':','-','--']):
        x=np.sort(arrays[sid]['importance'].ravel());y=np.arange(1,x.size+1)/x.size
        # Deterministic thinning changes only display resolution, not the ECDF ranks.
        ids=np.unique(np.linspace(0,x.size-1,min(900,x.size)).astype(int))
        ax.plot(x[ids],y[ids],color=col,linestyle=style,linewidth=1.5,label=sid)
    ax.set_xlabel('Recorded native-grid importance I');ax.set_ylabel('Within-image fraction of valid-content tokens')
    ax.set_ylim(0,1);ax.legend(ncol=3,fontsize=9);grid(ax,'both')
    finish(fig,'Importance distributions across the six saved samples',NOTE+'\nPadding removed; no cubic interpolation. Curves are descriptive and do not treat tokens as independent experiments.',bottom=.21,top=.85)
    save(fig,'Q10','六图原生token重要性累积分布','qualitative',boundary+'每图ECDF独立归一为1，展示至多900个确定性秩点，不拟合密度、不做token级显著性检验。',
         common_sources+[f'inputs/arrays/{s}/raw_importance.npy' for s in IDS],evidence='historical_diagnostic')

    fig,ax=plt.subplots(figsize=(9.4,5.1))
    for sid,col,style in zip(IDS,COLORS,['-','--','-.',':','-','--']):
        x=np.sort((arrays[sid]['corrected_energy']-arrays[sid]['input_energy']).ravel())
        y=np.arange(1,x.size+1)/x.size;ids=np.unique(np.linspace(0,x.size-1,min(900,x.size)).astype(int))
        ax.plot(x[ids],y[ids],color=col,linestyle=style,linewidth=1.5,label=sid)
    ax.axvline(0,color=C['ink'],linewidth=.8);ax.set_xlabel('Corrected feature RMS - input feature RMS')
    ax.set_ylabel('Within-image fraction of valid-content tokens');ax.set_ylim(0,1);ax.legend(ncol=3,fontsize=9);grid(ax,'both')
    finish(fig,'Distribution of recorded feature-energy changes',NOTE+'\nNative valid-content tokens; feature-energy changes are not a calibrated measure of noise removal.',bottom=.21,top=.85)
    save(fig,'Q11','特征能量变化的原生token累积分布','qualitative',boundary+'逐token计算同一历史前向已保存的两张能量图之差；不是新模型计算，也不是人为噪声实验。',
         common_sources+[f'inputs/arrays/{s}/raw_{k}.npy' for s in IDS for k in ['input_energy','corrected_energy']],evidence='historical_diagnostic')

    fig,axes=plt.subplots(2,3,figsize=(10.8,7.1))
    for ax,sid in zip(axes.flat,IDS):
        gt(ax,sid,False);ax.set_title(f'{sid} | {len(boxes[sid])} annotated objects',fontsize=10)
    finish(fig,'SAR scenes and official ground-truth boxes','Six fixed SSDD official TEST samples | native grayscale display: 0-255\nScene observations only: brightness is not a label of noise, clutter, or model reliability.',bottom=.14,top=.89)
    fig.subplots_adjust(left=.03,right=.97,wspace=.15,hspace=.28)
    save(fig,'Q12','真实SAR场景与官方GT框样本墙','qualitative',
         '六个样本全部展示，不按本轮结果挑选。仅解释场景和标注，不从高亮区域推出噪声、杂波或Mamba错误写入。图像与历史capture哈希相同，官方COCO框重新绘制。',
         ['inputs/ssdd_test_coco.json','inputs/historical_maps_manifest.json']+[f'inputs/images/{s}.jpg' for s in IDS],evidence='real_images')
