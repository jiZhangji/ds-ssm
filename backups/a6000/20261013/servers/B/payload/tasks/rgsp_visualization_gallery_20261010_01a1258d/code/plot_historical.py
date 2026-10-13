"""Separate panels for two historical multi-seed protocols and their limits."""
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from common import C, METRICS, LABELS, read_csv, finish, grid, color, label, LEGACY_NOTE, SADD_NOTE


def build(save, data_dir):
    root = Path(data_dir)
    protocols = [
        ("SADD", "sadd_3seed.csv", ["original", "matched_scale075", "local_energy", "full"],
         SADD_NOTE, 1, "sadd_multi_seed"),
        ("SSDD", "ssdd_legacy_individual.csv", ["baseline", "ds_only_p3", "pfim_only_p3", "full_p3", "global_p3"],
         LEGACY_NOTE+"\n*Legacy PFIM-only is not a strict independent PFIM control.", 5, "historical_multi_seed")
    ]
    for ds, fname, keys, note, offset, evidence in protocols:
        rows = read_csv(root/fname)
        groups = {k: sorted([r for r in rows if r["variant"]==k],key=lambda r:int(r["seed"])) for k in keys}
        if any([int(r["seed"]) for r in g] != [0,1,2] for g in groups.values()):
            raise ValueError("Expected exactly three observed seeds (0, 1, 2)")
        values = {k:np.array([[float(r[m])*100 for m in METRICS] for r in groups[k]]) for k in keys}
        if any(not np.isfinite(v).all() or np.any((v < 0) | (v > 100)) for v in values.values()):
            raise ValueError('Historical metrics must be finite rates in the 0-1 source scale')
        ap_idx = list(METRICS).index("ap50_95")
        ap = {k:v[:,ap_idx] for k,v in values.items()}
        source = ["inputs/"+fname]
        boundary = ("独立历史协议，保留全部3个seed。误差条是样本标准差(ddof=1)，不是置信区间或统计显著性。"
                    + ("SADD为阳性test44；恢复轨道非逐位等价；Original seed0是该12run中的53.7117，不能替换为当前seed0矩阵的54.0872。"
                       if ds=="SADD" else "旧SSDD每个seed同时改变数据划分和训练随机性，同seed内配对；样本SD包含两种来源的变异。"
                       "旧SSDD742/186/232已发现内容重叠；旧PFIM-only关闭参数漂移，不能作严格PFIM独立贡献；禁止与新清理划分混算。"))

        fig, ax = plt.subplots(figsize=(8.8,5))
        y = np.arange(len(keys))
        for i,k in enumerate(keys):
            mu,sd = ap[k].mean(),ap[k].std(ddof=1)
            ax.errorbar(mu,i,xerr=sd,fmt="D",color=color(k),markersize=7,capsize=4,linewidth=1.5)
            for seed, off in enumerate([-.16,0,.16]):
                ax.scatter(ap[k][seed],i+off,marker=["o","s","^"][seed],color=color(k),s=35,alpha=.7)
            ax.text(57.7 if ds=="SADD" else 50.8, i, f"{mu:.2f} +/- {sd:.2f}",va="center",ha="right",fontsize=9)
        ax.set_yticks(y,[label(k) for k in keys]);ax.invert_yaxis()
        ax.set_xlim((47,58) if ds=="SADD" else (36,51));ax.set_xlabel("Test AP50:95 (%)");grid(ax)
        handles = [Line2D([], [], color=C['gray'], marker=marker, linestyle='none',
                          markersize=5, label=f'Seed {seed}')
                   for seed, marker in enumerate(['o', 's', '^'])]
        handles.append(Line2D([], [], color=C['gray'], marker='D', linewidth=1.3,
                              markersize=5, label='Mean +/- sample SD'))
        ax.legend(handles=handles,ncol=4,loc='lower center',bbox_to_anchor=(.5,1.015),fontsize=8.5)
        finish(fig,f"{ds}: individual seeds and mean +/- sample SD",note,bottom=.2,top=.80)
        fig.subplots_adjust(left=.27,right=.98)
        save(fig,f"H{offset:02d}",f"{ds}三seed原值与均值标准差", "multiseed",boundary,source,evidence=evidence)

        reference=keys[0];others=keys[1:]
        fig,ax=plt.subplots(figsize=(9.2,5.6))
        for i,k in enumerate(others):
            delta=ap[k]-ap[reference]
            for seed,off in enumerate([-.19,0,.19]):
                ax.scatter(delta[seed],i+off,color=[C['blue'],C['orange'],C['purple']][seed],
                           marker=['o','s','^'][seed],s=60,label=f"Seed {seed}" if i==0 else None)
                ax.text(delta[seed]+(.08 if delta[seed]>=0 else -.08),i+off,f"{delta[seed]:+.2f}",
                        ha='left' if delta[seed]>=0 else 'right',va='center',fontsize=8.5)
            # The mean has a separate reading row, so its tick cannot cross
            # a seed's value label when their numerical values are close.
            ax.scatter(delta.mean(),i+.36,marker='|',s=180,linewidths=1.7,
                       color=C['ink'],zorder=5,label='Mean' if i==0 else None)
        ax.axvline(0,color=C['ink'],linewidth=.8);ax.set_yticks(range(len(others)),[label(k) for k in others]);ax.invert_yaxis()
        ax.set_xlim(-6.5,8);ax.set_xlabel("Within-seed AP50:95 difference from Original (percentage points)")
        ax.legend(ncol=4,loc='lower center',bbox_to_anchor=(.5,1.015),fontsize=8.5);grid(ax)
        finish(fig,f"{ds}: paired seed differences",note+"\nBlack tick below each seed group: mean. Same seed does not imply tensor-identical initialization.",bottom=.23,top=.80)
        fig.subplots_adjust(left=.27)
        save(fig,f"H{offset+1:02d}",f"{ds}逐seed配对差值与负结果", "multiseed",boundary+"配对按同seed索引，架构变化时不承诺逐张量同初始化。",source,evidence=evidence)

        metrics=list(METRICS)
        means=np.array([values[k].mean(axis=0) for k in keys]);sds=np.array([values[k].std(axis=0,ddof=1) for k in keys])
        fig,ax=plt.subplots(figsize=(10.5,4.8))
        im=ax.imshow(means,cmap='Blues',vmin=0,vmax=100,aspect='auto')
        for (i,j),v in np.ndenumerate(means):
            ax.text(j,i,f"{v:.2f}\n+/- {sds[i,j]:.2f}",ha='center',va='center',fontsize=9,
                    color='white' if v>67 else C['ink'])
        ax.set_xticks(range(len(metrics)),[METRICS[m].replace(' (%)','') for m in metrics]);ax.set_yticks(range(len(keys)),[label(k) for k in keys]);ax.tick_params(length=0)
        fig.colorbar(im,ax=ax,fraction=.035,pad=.025,label='Mean (%)')
        finish(fig,f"{ds}: five metrics across all seeds",note,bottom=.2,top=.85);fig.subplots_adjust(left=.24)
        save(fig,f"H{offset+2:02d}",f"{ds}五指标均值及标准差热力图", "multiseed",boundary+"各格上行为均值，下行为样本标准差，单位均为百分数。",source,evidence=evidence)

        fig,ax=plt.subplots(figsize=(9.2,5.3))
        for seed,marker,col in [(0,'o',C['blue']),(1,'s',C['orange']),(2,'^',C['purple'])]:
            v=[ap[k][seed] for k in keys]
            ax.plot(range(len(keys)),v,marker=marker,color=col,label=f'Seed {seed}',linewidth=1.4)
        ax.set_xticks(range(len(keys)),[label(k).replace(' (legacy)','\n(legacy)') for k in keys],rotation=0)
        ax.set_ylabel('Test AP50:95 (%)');ax.set_xlabel('Configuration (categorical; connecting lines aid within-seed reading)')
        ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.17));grid(ax,'y')
        finish(fig,f"{ds}: seed-to-seed changes across configurations",note,bottom=.23,top=.79)
        fig.subplots_adjust(left=.1,right=.97)
        save(fig,f"H{offset+3:02d}",f"{ds}逐seed配置折线剖面", "multiseed",boundary+"横轴是配置类别而非训练时间；连线不代表连续函数。",source,evidence=evidence)

    # Numbers parsed directly from the staged historical protocol table, not re-benchmarked.
    protocol=(root/'ssdd_legacy_protocol.md').read_text(encoding='utf-8-sig')
    complexity={}
    for line in protocol.splitlines():
        for k in ["baseline","full_p3","ds_only_p3","pfim_only_p3","global_p3"]:
            if re.match(r'^\|\s*'+k+r'[（(]',line):
                parts=[p.strip() for p in line.strip().strip('|').split('|')]
                complexity[k]={'params':int(parts[1].replace(',',''))/1e6,'gflops':float(parts[2])}
    if len(complexity)!=5:raise ValueError('Historical complexity table missing expected rows')
    keys=['baseline','ds_only_p3','pfim_only_p3','full_p3','global_p3']
    fig,axes=plt.subplots(1,2,figsize=(10.8,5))
    for ax,metric,title in zip(axes,['params','gflops'],['Parameters (million)','Recorded profiler GFLOPs']):
        vals=[complexity[k][metric] for k in keys]
        bars=ax.barh(range(5),vals,color=[color(k) for k in keys],height=.6)
        ax.bar_label(bars,labels=[f'{v:.3f}' if metric=='params' else f'{v:.2f}' for v in vals],padding=3,fontsize=9)
        ax.set_yticks(range(5),[label(k) for k in keys]);ax.invert_yaxis();ax.set_xlim(0,max(vals)*1.16);ax.set_xlabel(title);grid(ax)
    finish(fig,'Historical model-size and profiler ledger','Recorded 2026-10-01 | input 640 | not a new unified efficiency benchmark\nArchitecture shells differ; custom selective-scan FLOP coverage is not established here.',bottom=.21,top=.84)
    fig.subplots_adjust(left=.22,wspace=.75,right=.97)
    save(fig,'H09','历史模型参数量和计算量账本','efficiency',
         '从旧协议表直接提取；未运行profiler。计算量可能受自定义算子计数覆盖限制。不能把壳层及MLP变化引起的成本差异归为纯SSM机制，也不能外推当前冻结配置。',
         ['inputs/ssdd_legacy_protocol.md'],evidence='historical_ledger')

    old=read_csv(root/'ssdd_legacy_individual.csv')
    fig,ax=plt.subplots(figsize=(9.8,5.7))
    marker=['o','s','^','D','P']
    for idx,k in enumerate(keys):
        v=np.array([float(r['ap50_95'])*100 for r in old if r['variant']==k])
        ax.errorbar(complexity[k]['gflops'],v.mean(),yerr=v.std(ddof=1),fmt=marker[idx],
                    color=color(k),markersize=8,capsize=4,label=label(k),elinewidth=1.2)
    ax.set_xlim(12.98,13.77);ax.set_xlabel('Recorded profiler GFLOPs');ax.set_ylabel('Historical test AP50:95 (%)');grid(ax,'both')
    ax.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,-.2),fontsize=9)
    finish(fig,'Historical accuracy and compute trade-off','SSDD legacy protocol | AP: 3 split/training seeds, mean +/- sample SD\nProfiler ledger only; no fresh latency, FPS, or operator-coverage validation.',bottom=.31,top=.84)
    fig.subplots_adjust(left=.1)
    save(fig,'H10','历史精度与计算量权衡散点图','efficiency',
         '精度误差条为3个split/training seed的样本标准差，同时包含数据划分和训练随机性的变异；横轴为旧profiler GFLOPs而非时间或新测速。相同计算量点重叠是数据事实，不人为抖动成本坐标。',
         ['inputs/ssdd_legacy_protocol.md','inputs/ssdd_legacy_individual.csv'],evidence='historical_ledger')

    curvepath=root/'historical_training.csv'
    if curvepath.exists():
        raw=read_csv(curvepath)
        rows=[{k.strip():v for k,v in row.items()} for row in raw]
        epoch=np.array([float(r['epoch']) for r in rows])
        if len(epoch)!=300 or len(set(epoch))!=300:raise ValueError('Historical curve is not 300 distinct epochs')
        fig,ax=plt.subplots(figsize=(9.5,4.9))
        for key,col in [('metrics/mAP50(B)',C['blue']),('metrics/mAP50-95(B)',C['teal'])]:
            ax.plot(epoch,[float(r[key])*100 for r in rows],color=col,linewidth=1.4,label=key.replace('metrics/','').replace('(B)',''))
        ax.set_xlabel('Epoch');ax.set_ylabel('Recorded validation AP (%)');ax.set_ylim(0,100);ax.legend();grid(ax,'both')
        finish(fig,'Historical DDP-ex27 validation curves','Archived 300-epoch run | raw epoch records, no smoothing | not the current campaign',bottom=.18,top=.85)
        save(fig,'H11','历史300轮训练的验证集AP曲线','training',
             '旧DDP-ex27单次运行的逐轮原值，不是当前清理后SSDD/SADD的收敛曲线，不能据此比较新方法。验证集曲线不是test表现。',
             ['inputs/historical_training.csv'],evidence='historical_training')
        fig,axes=plt.subplots(1,3,figsize=(11.7,4.5))
        for ax,loss in zip(axes,['box_loss','cls_loss','dfl_loss']):
            for split,col,style in [('train',C['blue'],'-'),('val',C['orange'],'--')]:
                key=f'{split}/{loss}'
                ax.plot(epoch,[float(r[key]) for r in rows],color=col,ls=style,linewidth=1.2,label=split)
            ax.set_title(loss.replace('_',' '));ax.set_xlabel('Epoch');ax.set_ylabel('Recorded loss');grid(ax,'both')
        axes[0].legend()
        finish(fig,'Historical optimization diagnostics','Archived DDP-ex27 | 300 raw epochs | loss scales are not comparable across panels',bottom=.21,top=.80)
        fig.subplots_adjust(left=.07,wspace=.4)
        save(fig,'H12','历史训练与验证损失的分面曲线','training',
             '展示原始逐轮训练/验证box、cls、DFL损失，不补点、不平滑、不剔除异常轮。不同损失的纵轴尺度各自标注。',
             ['inputs/historical_training.csv'],evidence='historical_training')
