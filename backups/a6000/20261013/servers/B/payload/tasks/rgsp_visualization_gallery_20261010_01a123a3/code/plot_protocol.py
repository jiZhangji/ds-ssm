"""Protocol and evidence coverage diagrams, not synthetic model results."""
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from common import C, current_records, finish, grid, NEW_SOURCE


def build(save, data_dir):
    snapshot, records=current_records(data_dir)
    fig,axes=plt.subplots(1,2,figsize=(11,5.3),gridspec_kw={'width_ratios':[1.4,1]})
    splits=[('SSDD legacy',742,186,232),('SSDD cleaned',730,186,232),
            ('SADD positive-only',795,44,44),('Ship cleaned',31735,3970,3974)]
    for i,(name,*counts) in enumerate(splits):
        total=sum(counts);left=0
        for value,col in zip(counts,[C['blue'],C['orange'],C['teal']]):
            width=value/total*100
            axes[0].barh(i,width,left=left,height=.6,color=col)
            if width>12:axes[0].text(left+width/2,i,f'{width:.1f}%',ha='center',va='center',color='white',fontsize=9)
            left+=width
    axes[0].set_yticks(range(4),[s[0] for s in splits]);axes[0].invert_yaxis();axes[0].set_xlim(0,100)
    axes[0].set_xlabel('Share of retained images (%)')
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=c,label=l) for c,l in zip([C['blue'],C['orange'],C['teal']],['Train','Validation','Test'])],
                   ncol=3,loc='upper center',bbox_to_anchor=(.5,1.15),fontsize=9)
    axes[1].axis('off')
    table=axes[1].table(cellText=[[f'{n:,}' for n in s[1:]] for s in splits],
                         colLabels=['Train','Val','Test'],cellLoc='right',colLoc='center',bbox=[0,.05,1,.9])
    table.auto_set_font_size(False);table.set_fontsize(10)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('white');cell.set_facecolor(C['lightblue'] if r==0 else '#F4F7F9' if r%2==0 else 'white')
    finish(fig,'Dataset splits must stay separate','Retained image counts | image-level splits do not establish scene independence\nSSDD legacy results are not pooled with the cleaned protocol. Ship scores are unavailable in this snapshot.',bottom=.2,top=.8)
    fig.subplots_adjust(left=.22,right=.97,wspace=.18)
    save(fig,'P01','数据划分规模与协议边界','protocol',
         '四个已记录协议分别显示数量及比例，不把Ship未完成成绩补为0；新旧SSDD不混算，SADD仅阳性test44的限制保留。',
         ['inputs/project_state_snapshot.md','inputs/seed0_review.md'],evidence='protocol_metadata')

    names=['SSDD','SADD','Cross-task','SAR-ShipDataset','SAR-AIRcraft1.0']
    expected=[27,9,15,3,3];available=[20,9,1,0,0];receipt_only=[0,0,14,0,0]
    not_verified=[a-b-c for a,b,c in zip(expected,available,receipt_only)]
    if sum(expected)!=snapshot['coverage']['requested'] or sum(available)+sum(receipt_only)!=snapshot['coverage']['verified']:
        raise ValueError('Coverage totals differ from the frozen snapshot')
    fig,ax=plt.subplots(figsize=(10,5.5));y=np.arange(5)
    left=np.zeros(5)
    for vals,col,title in [(available,C['teal'],'Captured values'),(receipt_only,C['blue'],'Verified / values absent'),
                           (not_verified,'#E4E8ED','No complete receipt')]:
        ax.barh(y,vals,left=left,height=.6,color=col,label=title)
        for i,v in enumerate(vals):
            if v:ax.text(left[i]+v/2,i,str(v),ha='center',va='center',color=C['ink'] if col=='#E4E8ED' else 'white',fontsize=10)
        left+=vals
    for i,n in enumerate(expected):ax.text(n+.3,i,f'/{n}',va='center',fontsize=9)
    ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlim(0,30);ax.set_xlabel('Planned seed-0 positions (count)')
    ax.legend(ncol=3,loc='lower left',bbox_to_anchor=(0,1.02),fontsize=8.5)
    finish(fig,'Completion receipts and metric availability are different','Snapshot: 2026-10-10 10:07 CST | 44/57 receipts verified, 29 detection metric records\nCross-task: 15/15 receipts; only one complete metric record available locally.',bottom=.2,top=.79)
    fig.subplots_adjust(left=.22)
    save(fig,'P02','完成回执与可用数值的覆盖图','protocol',
         '只反映固定时点的回执与本地数值覆盖，不代表实时进度。14项跨任务没有数值不等于训练失败；AIR三项有数据协议冲突。',
         [NEW_SOURCE,'inputs/seed0_review.md'],evidence='snapshot_metadata')

    categories=['detection','reuse','command']
    cats={'detection':'Campaign run','reuse':'Reused result','command':'Historical adoption'}
    fig,ax=plt.subplots(figsize=(8.7,4.6));bottom=np.zeros(2)
    for kind,col in zip(categories,[C['blue'],C['teal'],C['orange']]):
        counts=np.array([sum(r['dataset']==ds and r['receipt_kind']==kind for r in records) for ds in ['SSDD','SADD']])
        ax.bar(['SSDD','SADD'],counts,bottom=bottom,width=.5,color=col,label=cats[kind])
        for i,n in enumerate(counts):
            if n:ax.text(i,bottom[i]+n/2,str(n),ha='center',va='center',color='white',fontsize=12)
        bottom+=counts
    ax.set_ylim(0,24);ax.set_ylabel('Recorded detection results (count)');ax.legend(loc='upper right');grid(ax,'y')
    finish(fig,'Where the 29 recorded results came from','Receipt categories from the fixed source JSON | not 29 wholly new uninterrupted runs',bottom=.18,top=.85)
    save(fig,'P03','最新检测结果的来源构成','protocol',
         '直接统计receipt_kind，分清新campaign、复用及历史采纳。没有将EMA恢复及旧审计缺口因重新整理而消除。',
         [NEW_SOURCE],evidence='snapshot_metadata')

    fig,ax=plt.subplots(figsize=(11.3,5.8));ax.set_xlim(0,11);ax.set_ylim(0,5);ax.axis('off')
    def box(x,y,w,h,text,fc):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04,rounding_size=.08',facecolor=fc,edgecolor='#C4CED7',linewidth=.8))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10,linespacing=1.5)
    box(.2,3.05,2.8,1.15,'Historical evidence\nSSDD 15 runs\nSADD 12 runs',C['lightorange'])
    box(4.1,3.05,2.8,1.15,'Preserve every seed\nKeep original protocols\nReport negative differences',C['lightorange'])
    box(8,3.05,2.8,1.15,'Historical panels\nH01-H12\nNo pooling with new trials',C['lightorange'])
    box(.2,.65,2.8,1.15,'Current seed-0 snapshot\nSSDD 20 / SADD 9\nCleaned SSDD protocol',C['lightblue'])
    box(4.1,.65,2.8,1.15,'Read verified test values\nKeep missing data absent\nDo not invent seed variance',C['lightblue'])
    box(8,.65,2.8,1.15,'Current panels\nC01-C16\nDescriptive single-seed results',C['lightblue'])
    for yy in [3.625,1.225]:
        for a,b in [(3.05,4.0),(6.95,7.9)]:
            ax.add_patch(FancyArrowPatch((a,yy),(b,yy),arrowstyle='-|>',mutation_scale=13,linewidth=1,color=C['muted']))
    ax.plot([.3,10.7],[2.42,2.42],color='#D7DDE4',linestyle=(0,(4,4)),linewidth=.8)
    ax.text(5.5,2.42,' separate evidence tracks ',ha='center',va='center',fontsize=10,color=C['muted'],bbox={'facecolor':'white','edgecolor':'none'})
    finish(fig,'An evidence-first comparison workflow','A diagram of reporting boundaries, not a model architecture or an experimental result',bottom=.08,top=.88)
    fig.subplots_adjust(left=.02,right=.98)
    save(fig,'P04','新旧协议隔离与可复现作图流程','protocol',
         '体现新旧证据轨道的报告边界。此图是工作流示意，不是模型结果；历史raw可追溯，当前数值缺口明确。',
         [NEW_SOURCE,'inputs/ssdd_legacy_protocol.md','inputs/seed0_review.md'],evidence='schematic')
