# RGSP 科研选图册

本包用于选图与后续精修，不替代项目正式状态。图内使用英文，选图说明和图注使用中文。

## 从哪里看

- 打开 `index.html`：按类别或图号筛选，可勾选并导出选图清单。纯离线，不需要启动网页服务。
- `contact_sheets/START_HERE.png`：12张不同用途的预览，不是按结果好坏筛选。
- `gallery_booklet.pdf`：50页连续浏览，每页一张图。
- `png/`：50张300 dpi图片；`svg/`：50张可编辑文字的SVG；`pdf/`：50张PDF。照片与特征栅格仍是栅格，文字、线条和结构框可编辑或矢量化。
- `manifest.json`：每图编号、标题、图注、来源、代码路径及文件SHA256。

## 选图建议

| 用途 | 建议先看 | 边界 |
|---|---|---|
| 最新总体结果 | C01、C02、C05、C16 | 单seed，多配置描述性比较，不写显著或稳定提升 |
| 主指标之外的差异 | C03、C04、C08、C09、C15 | C08是P/R工作点，不是PR曲线 |
| 机制消融 | C10–C14 | 正确四格及锚点，保留正负结果；不等于物理机制证明 |
| 三seed稳定性展示 | H01–H08 | SADD旧12run与SSDD旧15run分别成组，不能混入当前单seed表 |
| 精度和成本 | H09、H10 | 旧profiler账本，非本轮新测速；无统一FPS或完整算子计数承诺 |
| 训练过程 | H11、H12 | 独立历史DDP-ex27的300轮，不是当前实验收敛曲线 |
| 真实场景和热图 | Q01–Q12 | 历史ex27真实捕获，不是当前P3推理；所有6个固定样本都保留 |
| 架构和模块示意 | M01–M06 | 代码对齐示意，不以示意图充当实测证据 |
| 协议及覆盖情况 | P01–P04 | 固定时点，不是实时监控 |

可以直接反馈图号，例如“保留 C05、C10、Q07、M01；C10改成单栏，Q07缩为三图”。

## 证据快照与不可混合的实验

1. 当前检测数据固定为2026-10-10北京时间10:07:54–10:07:57的已回收快照：SSDD20项、SADD9项；五份回执合计44/57。活动训练可能继续，本包不会悄悄更新数字。
2. 当前SSDD使用730/186/232；旧SSDD15run使用742/186/232，已知内容重叠及旧PFIM-only冻结语义问题均保留。两个轨道禁止合并均值或排名。
3. SADD旧12run的Original seed0 AP50:95为53.7117%；当前矩阵Original seed0为54.0872%。这不是一条记录，不用后者替换前者。SADD仅44张阳性test图，不等于全场景或七类SAR-AIRcraft。
4. 误差条只在真实3seed记录上出现，定义为样本标准差（ddof=1），不是95%置信区间。旧SSDD每个seed同时改变数据划分和训练随机性，同seed内配对，因此SD包含两种来源的变异；SADD三seed共享固定split。无显著性星号、虚构seed、平滑补点或模拟研究结果。
5. 跨任务15项有完成记录，但当前来源仅回收1项完整指标，另14项数值缺失；因此不生成最新五数据集的三臂性能比较。Ship/AIR未回收结果也不填0。
6. 热图来自2026-08-23 exact_v2，使用历史ex27和block_index=0，不能当成当前P3模型的解释。六张原图与原manifest哈希一致，实际上属于官方test，旧目录名称`val`不改变这一事实。
7. Q01–Q07、Q10–Q11使用NPZ原生网格裁letterbox，不做双三次插值；Q08–Q09直接引用历史上裁letterbox后经双三次插值的aligned统计，不用本次官方COCO框重新计算分区均值。每种特征跨图使用统一色标，输入与修正能量共用色标。GT框并集不是分割mask，importance不是可靠性概率。
8. 不从SAR亮度推断噪声、不从特征能量下降推断去噪成功、不因保留了真实热图而声称机制已验证。

更详细的来源与对话可读范围见 `notes/evidence_inventory.md`、`notes/conversation_audit.md`。会话覆盖不等于恢复已删除、压缩或未保存的全部原文。

## 代码与复现

代码入口为 `code/build_gallery.py`。依赖只需Python 3.10+、NumPy、Matplotlib、Pillow；PDF版面核验另需系统Poppler。生成代码没有导入torch、加载权重或发起网络请求。

请依项目规则先选择合适服务器，在独立目录解压本包后执行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  python code/build_gallery.py --out . --dpi 300
python code/verify_outputs.py --out .
```

不需要重新训练或推理。`inputs/`含生成所需65份原样副本或ZIP成员及`source_manifest.json`，生成前核对全部输入哈希。

`code/stage_inputs.ps1`用于在原工作区重新收集输入：仅复制小表、源码证据、6张官方图像、COCO标注及NPZ中的raw数组，不修改原始数据或实验；默认SSDD ZIP路径记录在脚本参数，可自行指定。若数据更新，应新建一个快照包，不覆盖本包的原始数字。

各模块：

| 脚本 | 图号 |
|---|---|
| `plot_current.py` | C01–C16 |
| `plot_historical.py` | H01–H12 |
| `plot_qualitative.py` | Q01–Q12 |
| `method_diagrams.py` | M01–M06 |
| `plot_protocol.py` | P01–P04 |
| `common.py` | 统一样式、标签与输入量纲检查 |
| `build_gallery.py` | 批量导出、选图页、PDF合册和图片清单 |
| `verify_outputs.py` | 文件/哈希/PDF字体和Poppler渲染核验 |

SVG保留可编辑文字，显示端请安装DejaVu Sans或选择相近字体；需要严格保持版面时使用已嵌入字体的PDF或PNG。

## 本轮运行边界

本轮只整理、重绘已记录证据并输出示意图，不启动训练、推理、基准或研究模拟，不安装软件，不改变原队列、服务、模型代码、原论文或原数据。服务器选择和产物验收见 `notes/EXECUTION_AND_QA.md`。

科研写作与图型参考：Leey21 的 [awesome-ai-research-writing](https://github.com/Leey21/awesome-ai-research-writing)，取其“论文架构图”“实验绘图推荐”“Reviewer审视”的设计原则，具体改写见 `notes/PROMPTS_AND_DESIGN.md`。网站提示词不构成数据来源或研究结论。
