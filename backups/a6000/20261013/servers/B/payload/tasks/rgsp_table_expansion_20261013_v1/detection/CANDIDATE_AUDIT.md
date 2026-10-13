# 检测候选来源与现成可运行性审计

日期：2026-10-12。状态：`STATIC_AUDIT_ONLY`。

本报告只读取本地项目文件与已经保存的结果记录；未 SSH、未联网核查上游、未导入模型、未运行测试/训练/推理，也未检查服务器实时文件或负载。历史成功记录不等于当前服务器可运行。本文是独立任务证据，不是第二套正式项目状态；正式 MemoryRoot 仍为 `docs/project-memory/`。

## 1. 范围与总判断

本轮名单以 `experiments/rgsp_table_completion_20261012_v1/CANDIDATE_PRIORITY.csv` 的 seed0 行为准，以 `experiments/paper_table_fill_20261012_v1/input/main.tex` 核对方法身份和表注。不同版本的旧完整稿、局部检测片段和全图稿不能合并成一份新名单。

| 数据集 | 外部方法位置 | 本地方法位置 | 合计 | 边界 |
|---|---:|---:|---:|---|
| SSDD | 17 | 3 | 20 | YOLOv8n 在名单内，YOLOv8s 不在 |
| SADD | 18 | 3 | 21 | YOLOv8n、YOLOv8s 都在；不能沿用阳性子集结果冒充全量协议 |
| SAR-ShipDataset | 23 | 3 | 26 | 未标规模的 YOLOv8/v10/11 仍保持未标规模 |
| 合计 | 58 | 9 | 67 | 去重后 42 种外部方法、3 种本地方法 |
| SAR-AIRcraft-1.0 | 不在当前 CSV | 不在当前 CSV | 不计入 67 | 第 6 节保留历史 AIR 表来源；不能用 SAR-ACD 分类行替代 |

以上仅为检测子集。PLAN 的 110 配置/220 个两-seed 位置包含分类与变化检测，不能写成 220 项现成检测训练。

静态证据能确认的最近一套统一训练入口只支持 SSDD 的 `original`、`matched_scale075`、`full`、`cnn_yolov8n` 四臂；YOLOv8n 具有该旧任务完整 300 epoch/8700 update 的保存回执。其余外部方法即使已有框架、源码或复杂度结果，也没有在已审证据中取得本轮统一训练/数据适配/重载评估验收。本轮协调者另行实现 Original/Matched075/Full/YOLOv8n/YOLOv8s；本报告不替代其新适配器验收，也不将 YOLOv8s 加入 SSDD 原候选矩阵。

## 2. 如何理解“有代码”和“可运行”

本文分开记录四层证据：

1. **来源**：原论文、作者仓库或框架官方实现已定位。论文数值的转引来源与原方法来源分开。
2. **现成 repo/config**：本地保存记录证明某个服务器目录及配置曾存在；没有现场复查时一律是历史路径。
3. **环境**：实际结果写明的版本才标已知；只有环境名称不推断 torch/mmcv/CUDA 等版本。不同服务器、不同隔离环境不得互相借用结论。
4. **统一 adapter**：须真正连接本轮数据清单、训练预算、优化器、损失、选模和评估，并通过服务器端保存/重载/有限性检查。profile 的 `forward` 能运行不证明训练 backward、loss 或完整解码可用。

表内状态：`OLD_RUNNER`=旧任务统一训练入口与证据；`PROFILE_ONLY`=已有构建/复杂度证据但无本轮训练 adapter；`IDENTITY_BLOCKED`=模型规模或完整配置未绑定；`SOURCE_INCOMPLETE`=仅部分作者代码/诊断重建；`UPSTREAM_ONLY`=只有公开上游定位线索；`UNKNOWN`=本次已审材料缺证据。`UNKNOWN` 不等于证明代码不存在。

下文标记 **U** 的公开上游 URL 是供继续核验的定位线索，未在本轮联网访问或锁定快照；不能据此声称已经下载、安装或匹配了文献成绩。已保存源码下载脚本/原始结果明确指向的来源则标 **E**。本报告没有把任何外部配置标为本轮 `READY_TO_TRAIN`。

## 3. 环境、repo 与配置证据

### E11：A6000-F，2026-10-11 复杂度任务

来源：`reports/SSDD_COMPLEXITY_20261011.md`、`complexity_benchmark/*20261011.py`、`artifacts/complexity_20261011/*results.json`。

保存报告给出任务根 `/home/powerleader/codex_tasks/complexity_20261011` 和隔离目录 `env`、`env_legacy`，明确未改现有训练环境、未训练或评估检测精度。下表路径由报告任务根与实际 profile 源码拼接；是可追踪的历史映射，不是本轮 `ls` 回执。两个隔离目录的 torch/mmcv/CUDA 精确版本、各脚本当时究竟调用哪一个 Python，在本次已审结果中没有完整绑定，保持 `UNKNOWN`；不能把 E12 的 torch 1.13.1 移植到它们。

同轮协调者另行转述：其父任务已通过 SSH 看到 F 的 `complexity_20261011/{env,env_legacy,YOLOX-main,FBGBNet-master,mmdetection-2.25.2}`，未核其余全部目录。该信息暂为 `COORDINATOR_REPORTED`：本审计未取得并独立读取原始回执路径，不能升级为本审计的现场确认；更不证明版本、源码完整性、训练 adapter 或 GPU 就绪。待父任务给出原始证据后可定点补充，不回写成所有候选可运行。

| 历史任务相对路径 | 已保存的实现/配置证据 | 已知版本与限制 |
|---|---|---|
| `yolov5-7.0/` | `models/yolov5n.yaml`、`models/yolov5s.yaml`、`models/hub/yolov3.yaml` | 官方 YOLOv5 tag v7.0；无本轮 repo SHA。YOLOv3 是该框架 hub 实现，不是已经绑定原 Darknet run |
| 环境内 `mmdet/.mim/configs/` | `fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py`；`cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py` | 报告记 MMDetection 3.3.0；实际 site-packages 绝对路径未保存。ROIAlign/NMS 等计数缺口保留 |
| `YOLOX-main/` | nano/tiny/s/m/l/x 六种规模单类 profile | 作者 main；提交/完整训练环境 `UNKNOWN`；候选表未指明规模 |
| `CenterNet-master/` | `models/CenterNet-52.py`、`models/CenterNet-104.py` | Keypoint Triplets；用方向 cummax 代替旧角点池化扩展，关闭解码统计网络 forward；不是训练入口验收 |
| `FBGBNet-master/` | `configs/vfnet/vfnet_r50_fpn_1x_coco.py` | FGBG-Net 作者仓库拼写是 FBGBNet；任务副本补 19 个 MMDetection 文件，并删两个未用 import；不是未修改原始作者快照 |
| `mmdetection-2.25.2/` | FGBG 框架补文件与 FEPS 诊断来源 | tag v2.25.2；不等于 E11 现代 mmdet 3.3.0 可混用 |
| `FEPS-Net-main/FEPS.py` | 作者 neck；另拼 RetinaNet R50 与兼容 DCN | 非作者确认的完整 detector/config；只能诊断，不能标 FEPS-Net 复现 |
| `MaDiNet-master/` | 下载脚本与报告指向作者 master；作者核心 `MambaSAR.py` | 完整扩散 detector 集成/config 未确认；不能用 res4+res5 组件复杂度替代全模型 |

源码定位：`fetch_sources_20261011.py`（YOLOX、FBGBNet、MaDiNet、FEPS-Net）、`fetch_small_sources_20261011.py`（补查 MaDiNet/FEPS/CenterNet）、`profile_official_yolov5_20261011.py`（v7.0）。对只有默认分支的 ZIP，未保存 commit 的地方不杜撰 commit。

### E07：3090 历史完整 detector 复杂度，不是 F 现成环境

来源：`reports/detector_complexity_evidence_20260724.md` 与 `artifacts/complexity_20260724/results/`。记录是服务器 **CPU FP32** profile，torch `2.1.1+cu121`、Detectron2 `0.6`，根 `/home/zhangji/codex_complexity_20260724/repos`。不能误写成 A6000-F 的 GPU 训练验收。

| 方法 | repo 相对目录 / config | 保存的 source commit |
|---|---|---|
| Faster R-CNN R50-FPN | `detectron2_official/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml` | `02b5c4e295e990042a714712c21dc79b731e8833` |
| RetinaNet R50-FPN | `detectron2_official/configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml` | `02b5c4e295e990042a714712c21dc79b731e8833` |
| DiffusionDet R50-FPN | `DiffusionDet/configs/diffdet.coco.res50.yaml` | `1efb36d4b0b721f2d509bc72656235a94d73ab02` |
| ConsistencyDet R50-FPN | `ConsistencyDet/configs/consisdet.coco.res50.yaml` | `0f6e44a260f70f10c7632b9da56cb755bffe2cc8` |
| DiffDet4SAR R50-FPN | `DiffDet4SAR/configs/diffdet.coco.res50.300boxes.yaml` | `0baf842ab7d33959792a7ed7b58f8cebc6a8f175` |

三种生成式 detector 均记录 500 proposals；DiffDet4SAR 文件名含 300boxes 不能覆盖实际配置字段。其原 override YAML 不合法，profile 按原文件显式字段重建；ConsistencyDet 修改了硬编码 CUDA 转移以跟随张量设备，其实际 forward 为 40 次 head，而不是看到 `sample_steps=1` 就当一次。Faster/Retina 的 R50-FPN 是结构参考，未绑定 MaDiNet 文献基线的原训练配置。其他三种也没有本轮统一数据/训练回执。

### E12：2026-10-12 首批 SSDD 已保存训练通路

来源：`experiments/rgsp_table_completion_20261012_v1/training/README.md`、`train_ssdd_v2.py:29`、`evidence/startup_f.json`、`evidence/completed_yolov8n_20261012.json`。

- 任务根：`/home/powerleader/ig-ds-ssm/tasks/rgsp_table_completion_20261012_v1/training`；冻结代码 `frozen_code/`。
- 保存的 F/D Python：`/home/powerleader/ig-ds-ssm/envs/ig-ds-ssm-env/bin/python`；实际 torch `1.13.1+cu117`。冻结清单 350 项，由 `frozen_source_hashes.json` 绑定。这里的环境证据不外推到 E11 的 `env/env_legacy`。
- `train_ssdd_v2.py` 明确只接受 Original/Matched075/Full/YOLOv8n；仅处理单类、928 train / 232 TEST-as-VAL。不能拿此入口直接声称 SADD 全量、Ship 或七类 AIR 已适配。
- YOLOv8n：该旧任务已有 300 epoch/8700 updates、FP32 最佳权重重载与完整回执。Original/Matched075/Full：已保存短测/启动/进展，不由本报告推断现在是否完成。
- 新任务的 YOLOv8s 以及通用数据/recipe 适配由协调者实现和验收；本审计未执行它们。

## 4. 完整外部候选审计（42 种）

数据集缩写：D=SSDD，A=SADD，S=SAR-ShipDataset。下列 membership 严格来自当前 CSV，而不是历史文献在哪些数据集出现过。表内“无 adapter”均指本次已审证据没有本轮统一训练适配器验收，不表示作者项目绝对没有 train.py。

### 4.1 有 repo / 框架或既有 profile 证据的候选

| 方法 | 表 | 来源 | 现成证据及本轮可运行性判断 |
|---|---|---|---|
| Faster R-CNN | D,A,S | Detectron2 官方框架参考；[上游 U](https://github.com/facebookresearch/detectron2) | E07 R50-FPN/config/commit 已知；文献身份未精确绑定，F 环境及本轮 adapter 未证实。`PROFILE_ONLY` |
| RetinaNet | D,S | 同上 Detectron2 官方框架参考 | E07 R50-FPN/config/commit 已知；无本轮 adapter，不能把其他头/参数量当该模型。`PROFILE_ONLY` |
| Cascade R-CNN | D,A,S | [MMDetection 官方框架 U](https://github.com/open-mmlab/mmdetection) | E11 MMDet 3.3.0 R50-FPN 前向 profile；训练入口/数据转换未验。`PROFILE_ONLY` |
| FCOS | D,S | MMDetection 官方框架参考 | E11 caffe-R50/FPN/GN-head profile；不是 Improved FCOS。`PROFILE_ONLY` |
| YOLOv3 | D,S | [YOLOv5 v7.0 E](https://github.com/ultralytics/yolov5/tree/v7.0) 的 hub YOLOv3 | E11 现成源码/单类 profile；需明确这一路实现而非项目现代 v3 或未核 Darknet 权重。无本轮 adapter。`PROFILE_ONLY` |
| YOLOv5n | D,S | [官方 v7.0 E](https://github.com/ultralytics/yolov5/tree/v7.0) | E11 `yolov5n.yaml` 已 profile；不可换成现代 Ultralytics v5 YAML，其结构/参数量不同。无本轮 adapter。`PROFILE_ONLY` |
| YOLOv5s | D | 官方 v7.0，同上 | E11 `yolov5s.yaml` 已 profile；不自动增加 Ship 的 v5s 行。无本轮 adapter。`PROFILE_ONLY` |
| YOLOv8n | D,A | [Ultralytics E](https://github.com/ultralytics/ultralytics)；项目冻结 n 配置 | E12 在旧 SSDD 协议完整跑通；SADD 新全量和本轮新协议未由该回执证明。`OLD_RUNNER`，不是全部数据集 READY |
| YOLOv8s | A | Ultralytics，同上 | 规模 s 明确，但旧 E12 CLI 无 s；本轮由协调者实现。不能加入 SSDD 原候选。旧服务器训练验收 `UNKNOWN` |
| CenterNet | D,S | [Duankaiwen/CenterNet E](https://github.com/Duankaiwen/CenterNet) | E11 Keypoint Triplets 的 HG52/HG104 可 profile；不是 Objects as Points。文献未绑定 HG 规模，且 pool/decode 有诊断替换。`IDENTITY_BLOCKED` |
| YOLOX | D | [Megvii-BaseDetection/YOLOX E](https://github.com/Megvii-BaseDetection/YOLOX) | E11 nano/tiny/s/m/l/x 均曾 profile，但原行规模未知；不能择一替代。`IDENTITY_BLOCKED` |
| DiffusionDet | D,S | [ShoufaChen/DiffusionDet E](https://github.com/ShoufaChen/DiffusionDet) | E07 完整 R50-FPN 构建/CPU profile；F 现成 repo/env 未证明，统一 runner 无证据。`PROFILE_ONLY` |
| ConsistencyDet | D,S | [Tankowa/ConsistencyDet E](https://github.com/Tankowa/ConsistencyDet) | E07 已修设备转移的 profile；40-head 成本及实际采样流程须保存。无本轮 runner。`PROFILE_ONLY` |
| DiffDet4SAR | D,S | [JoyeZLearning/DiffDet4SAR E](https://github.com/JoyeZLearning/DiffDet4SAR) | E07 单/七类 profile 可追踪；作者重点 AIR，SSDD/Ship 原配置不完整，YAML 修复必须披露。无本轮 runner。`PROFILE_ONLY` |
| FGBG-Net | D | [Xidian-AIGroup190726/FBGBNet E](https://github.com/Xidian-AIGroup190726/FBGBNet) | E11 修补任务副本 R50/VFNet profile，未绑定原论文成绩版本；训练/多分支行为未验。`PROFILE_ONLY` |
| FEPS-Net | D | [so-bright/FEPS-Net E](https://github.com/so-bright/FEPS-Net)；[论文](https://doi.org/10.1109/JSTARS.2022.3230859) | 作者只公开 neck/依赖调用；本地 profile 是 RetinaNet+兼容 DCN 重建，非完整原方法。`SOURCE_INCOMPLETE` |
| MaDiNet | D,S | [JoyeZLearning/MaDiNet E](https://github.com/JoyeZLearning/MaDiNet)；[作者稿](https://arxiv.org/abs/2411.07500v1) | E11 仍无确认完整 detector/config；既有 res4+res5 仅组件。`SOURCE_INCOMPLETE`，不能直接以 DiffusionDet 替代 |
| PEGNet | D | [原论文](https://doi.org/10.1109/LGRS.2025.3528221) | E11 明确未确认完整作者实现、全文受限；同名图网络/区块链仓库不是该方法。repo/env/adapter `UNKNOWN`，`IDENTITY_BLOCKED` |

### 4.2 通用上游明确，但本次没有对应完整运行链的候选

这些上游不能通过“已经有 Ultralytics”或“MMDetection 支持”自动升级为本轮可运行。尤其新版本 YOLO 不能默认存在于冻结版框架中。

| 方法 | 表 | 来源线索 | 身份 / 环境 / adapter 缺口 |
|---|---|---|---|
| YOLOv4 | A,S | [论文 E](https://arxiv.org/abs/2004.10934)；[AlexeyAB/darknet U](https://github.com/AlexeyAB/darknet) | 原表配置/权重、服务器 repo/env、统一 adapter 均未绑定；不能仅按名称选择任意 v4 规模。`UPSTREAM_ONLY` |
| YOLOv7 | A,S | 当前稿 CVPR 2023 书目；[WongKinYiu/yolov7 U](https://github.com/WongKinYiu/yolov7) | 标准/x/其他配置未由原表训练链锁定；无已审服务器安装或统一 runner。`UPSTREAM_ONLY` |
| YOLOR-P6 | A | YOLO-SAD Table 9 转引；[WongKinYiu/yolor U](https://github.com/WongKinYiu/yolor) | P6 名称已定，但 repo commit、数据/训练配置、环境与 adapter `UNKNOWN`。`UPSTREAM_ONLY` |
| YOLOv10n | A | MSAD Table 3 转引；[THU-MIG/yolov10 E](https://github.com/THU-MIG/yolov10)（旧复现审计已登记） | n 明确；端到端头/损失/后处理需原实现 adapter，不能当普通 YOLOv8 loss。无安装/运行证据。`UPSTREAM_ONLY` |
| YOLO11n | A | MSAD Table 3 转引；[Ultralytics U](https://github.com/ultralytics/ultralytics) | n 明确；对应支持版本、commit、服务器环境未知，无本轮 adapter。`UPSTREAM_ONLY` |
| YOLOv8 | S | [TPNet Table 4](https://doi.org/10.3390/rs17091487) 转引；Ultralytics 上游 | 原文表名不含规模，当前稿明说未自行指定；不能以 n/s 替代。`IDENTITY_BLOCKED` |
| YOLOv10 | S | TPNet Table 4 转引；THU-MIG/yolov10 上游 | 原行规模 `UNKNOWN`；不能以 SADD 的 v10n 代替。`IDENTITY_BLOCKED` |
| YOLO11 | S | TPNet Table 4 转引；Ultralytics 上游 | 原行规模 `UNKNOWN`；不能以 SADD 的 11n 代替。`IDENTITY_BLOCKED` |
| PP-YOLO | S | TPNet Table 4 转引；[PaddleDetection U](https://github.com/PaddlePaddle/PaddleDetection) | 当前稿无原论文书目/official-code 引用；backbone/规模未绑定，无 Paddle 环境/adapter 证据。`IDENTITY_BLOCKED` |
| PP-YOLOv2 | S | 同上 TPNet / PaddleDetection U | 不从名称推断 R50/R101、预训练与配方；repo/env/adapter 未验。`IDENTITY_BLOCKED` |
| PP-YOLOE | S | 同上 TPNet / PaddleDetection U | s/m/l/x 及具体变体 `UNKNOWN`；不能换为名字接近的 PyTorch 实现。`IDENTITY_BLOCKED` |

### 4.3 论文存在，但完整官方实现尚未在证据中确认的候选

| 方法 | 表 | 可核对的论文 / 数值来源 | repo、环境与统一 adapter 判断 |
|---|---|---|---|
| ResNet50-CSP | A | [YOLO-SAD](https://doi.org/10.3390/app14073025) Table 9 二次基线 | 完整 detector/config、原始出处/年份未核；不是 torchvision ResNet-50 分类器。全部 `UNKNOWN` |
| ResNeXt50-CSP | A | YOLO-SAD Table 9 | 同上；不能以 ResNeXt50 backbone 代替整个候选。全部 `UNKNOWN` |
| SEFEPNet | A | [FEMSFNet](https://doi.org/10.3390/rs16091589) Table 4 转引原 SADD 基线 | 原始论文/作者代码 URL 在当前证据 `UNKNOWN`；不能从相关名称猜仓库。环境/adapter `UNKNOWN` |
| YOLO-SAD | A | [原论文](https://doi.org/10.3390/app14073025)，Table 9 | 未确认作者完整代码/config；不能当通用 YOLOv7 的别名。repo/env/adapter `UNKNOWN` |
| FEMSFNet | A | [原论文](https://doi.org/10.3390/rs16091589)，Table 4 | 已知文献使用含负样本、5:1、224 输入；未确认完整 repo/config/env/adapter。`UNKNOWN` |
| MSAD-T | A | [原论文](https://doi.org/10.1049/cvi2.70032)，Table 3 | T 规模明确；模型实现/训练配置未绑定；不能以 Mamba-YOLO-T 代替。repo/env/adapter `UNKNOWN` |
| MSAD-B | A | 同上 MSAD，Table 3 | B 规模明确但完整实现未确认；历史 AIR 报告也不构成当前可运行证据。repo/env/adapter `UNKNOWN` |
| CCS-Net | A | [原论文](https://doi.org/10.3390/aerospace13090819)，Table 6 | 已审稿只给论文，不提供确认的作者代码/版本；repo/env/adapter `UNKNOWN` |
| WSF-YOLO | A | [原论文](https://doi.org/10.3390/rs18183254)，Table 12 | 波域/频域修改不能用通用 YOLO 伪装；repo/env/adapter `UNKNOWN` |
| Improved FCOS | S | [原论文](https://doi.org/10.1109/JSTARS.2022.3213583)；数值来自 MaDiNet v1 Table 4 | 旧审计未找到完整公开实现；是 SAR 改进而非 E11 通用 FCOS。repo/env/adapter `UNKNOWN` |
| CMFT | S | [原论文](https://doi.org/10.1109/TGRS.2023.3300536)；数值来自 MaDiNet v1 Table 4 | 跨模态特征迁移；完整源码、所需模态/预训练资产、训练权限均未绑定。repo/env/adapter `UNKNOWN` |
| LRTransDet | S | [原论文](https://doi.org/10.3390/rs15225309)，自身 Ship 实验 | 旧静态审计未确认完整官方代码；不能仅按轻量 ViT 结构重建后冒充。repo/env/adapter `UNKNOWN` |
| TPNet | S | [原论文](https://doi.org/10.3390/rs17091487)，Table 4 | 未确认完整作者 repo/config/env/adapter；其比较表的 YOLO 名称也不能决定具体规模。`UNKNOWN` |

### 4.4 本地三臂与旧训练回执

Original、Matched075、RGSP Full 在 D/A/S 各占一个候选位置，共 9 个位置。实现身份由项目冻结 manifest 与控制配置绑定，不以旧 ex27、PFIM-T/S 或 LocalEnergy 替代当前 Full。E12 只证明三臂旧 SSDD 通路已做短测并启动；旧 SADD 阳性子集和旧 Ship 150 epoch 结果各有独立来源，不能填入新协议行。新任务是否可跑以协调者当前 adapter + 服务器 probe + data readiness 为准。

## 5. 文献来源分组与不能自动放行的条件

- SSDD：一般外部行来自 MaDiNet 作者稿 Table 3；FEPS/FGBG/PEG 的数值由 SARDet-MIM Table 1 转引。不同输入/算子覆盖的复杂度即使已经填表，也不证明原成绩模型完全绑定。
- SADD：YOLOv4、ResNet50-CSP、ResNeXt50-CSP、YOLOR-P6、YOLOv7、YOLO-SAD 来自 YOLO-SAD Table 9（701 去重图、60:40）；SEFEPNet/FEMSFNet 来自 FEMSFNet Table 4；Faster/Cascade、YOLOv8n/s、v10n、11n、MSAD-T/B 来自 MSAD Table 3；CCS-Net 和 WSF-YOLO 分别来自各自 Table 6/12。不能把这些不同协议结果混成统一跑通的证据。
- Ship：15 行来自 MaDiNet v1 Table 4，LRTransDet 来自自身实验；PP-YOLO、PP-YOLOv2、PP-YOLOE、YOLOv8、YOLOv10、YOLO11、TPNet 共 7 行来自 TPNet Table 4。泛 YOLO 行必须先决定可追溯规模，再作为新统一配置登记，不能回写为已复现原行。
- 本次核读的主稿有若干缺失原方法书目键以及陈旧正文说明；这降低来源链完备程度，不影响按 CSV 保留候选。缺书目不靠猜测修补，本审计未修改论文。
- 所有方法放行前都要绑定完整配置/上游版本/环境、data manifest、类别数、框型与输入变换、原生 loss、统一预算和 evaluator。需要新安装/重负载的操作不由本报告自动授权。

## 6. SAR-AIRcraft 历史表：独立保留，不加入当前 67 配置

当前补表主稿没有独立 AIR 检测表，`CANDIDATE_PRIORITY.csv` 也没有 AIR。为避免遗漏已知历史对比，以下完整登记 `reports/four_dataset_methods_metrics_opensource_audit_20260823.md` §5.1/5.2 的 18+8 个外部名称。其“开源”等级仅是当时文献/仓库审计，不能当今日本地/服务器安装证据；旧文献页/仓库本轮未联网重验。

| 历史 AIR 方法 | 原报告的来源情况 | 本轮判定 |
|---|---|---|
| SKG-Net、SA-Net、MLSDNet | MaDiNet 作者稿 AIR Table；旧审计未找到官方完整代码 | 原方法 repo/env/adapter `UNKNOWN`；不以近似 detector 替代 |
| FCOS、RetinaNet、Cascade R-CNN、Faster R-CNN | 通用框架有公开实现 | E07/E11 只部分构建/单类或指定类数 profile；七类 AIR 新训练 adapter 未验 |
| RepPoints | 通用 MMDetection 框架实现线索 | 没有该候选的本次已审 repo/config/profile/adapter 回执 |
| CenterNet | 公开通用实现，E11 已澄清 Keypoint Triplets | AIR 原行 HG 规模、协议与训练入口仍未绑定 |
| YOLOX | 公开作者实现 | AIR 原行规模未绑定；E11 单类 profile 不证明七类 AIR 训练已就绪 |
| YOLOv3、YOLOv5n、YOLOv8n | 通用公开实现 | 有其他数据集 profile/旧训练证据，不是 AIR 新协议结果 |
| ConsistencyDet、DiffusionDet、DiffDet4SAR | 作者仓库；E07 有 7 类 JSON | 仅历史 CPU forward/profile，未确认本轮 AIR 数据/训练/统一 evaluator；DiffDet4SAR 作者确有 AIR 说明但不能绕过数据冲突 |
| MaDiNet | 作者仓库不完整 | 仍缺完整 detector/config/checkpoint 绑定 |
| SADRN | [集成检测识别论文](https://doi.org/10.3390/rs16132471) | 原审计未找到确认官方代码；不能混用 MaDiNet 的评估口径 |
| SFS-CNet | [作者代码 E](https://github.com/like413/SFS-Conv) | 旧审计称官方代码但精简；服务器 repo/env/完整 AIR adapter `UNKNOWN` |
| PGD-YOLOv8 | [作者代码 E](https://github.com/XAI4SAR/PGD) | 旧审计记代码/预训练权重；本轮未回收或锁版本，初始化与 scratch 不可混用 |
| MSAD-B | [原论文](https://doi.org/10.1049/cvi2.70032) | 原审计未找到完整作者代码；SADD 与 AIR 不能视为同一数据集 |
| SARDet-MIM | [原论文](https://doi.org/10.3390/rs18040580) | 预训练/结构配置与本轮 scratch 不同，repo/env/adapter 未确认 |
| MSPaDet | [原论文](https://doi.org/10.3390/rs18101513) | MMRotate/指标口径与 HBB AP50 不同；原审计无确认完整代码 |
| DenoDet、DenoDet V2 | [GrokSAR E](https://github.com/GrokCV/GrokSAR) | 旧审计记 config/log/checkpoint；本轮服务器/版本/adapter未验，VOC07/VOC12/COCO 指标不得只按数值大小合并 |

本地历史 AIR 对照还包含 Mamba-YOLO-T reference VAL/TEST、PFIM-T VAL/TEST、PFIM-S VAL 和更早 PFIM-T 档案；它们不是当前 Original/Matched075/Full 的同协议重跑，不在上述外部计数中。现行正式记忆 CONFLICT-017 保留所审 test 内 797 组语义冲突及三个阻塞位置；本报告不修复/裁决标签、不减小旧 57 项门槛、不放行旧 seed1。

## 7. SSD、DETR、Deformable DETR、EfficientDet、YOLOv9/12、RF 等名称

这些名称不出现在当前 SSDD/SADD/Ship 候选 CSV，也不在该补表稿三张外部检测表中。旧历史可能在其他数据集/全图稿出现 SSD、YOLOv9-tiny、YOLO12n，但不是当前矩阵的新授权或已经实现的候选。

| 名称 | 当前范围 / 可运行证据 |
|---|---|
| SSD | 不在当前三检测表/CSV；不能混淆 SSD detector、SSD-YOLO 和 Mamba-2 的 Structured State Space Duality。具体 SSD300/512、backbone、实现未绑定 |
| DETR | 不在当前名单；不能以现有框架中的 RT-DETR 当作原 DETR。repo/env/adapter `UNKNOWN` |
| Deformable DETR | 不在当前名单；多尺度可变形注意力依赖/config/env/adapter 未审 |
| EfficientDet | 不在当前名单；D0–D7/实现框架均未指定，未审服务器运行链 |
| YOLOv9 / YOLOv9-tiny | 旧历史其他表有名称，当前矩阵没有；不能临时增入或视为 YOLOv5 兼容别名 |
| YOLOv12 / YOLO12n | 旧历史其他表有名称，当前矩阵没有；当前冻结环境支持情况 `UNKNOWN` |
| RF / RF-DETR | 当前名单无对应项；“RF”本身身份含糊，不能直接解释成某个 RF-DETR 规模并启动 |

如协调者后续把其中某项纳入新任务，需另存明确的新候选身份与范围决定，然后补 repo/config/env/adapter/data/probe 证据；本报告只记缺口，不静默扩张原候选。

## 8. 审计覆盖与来源指纹

静态核对：67 个 seed0 检测位置、58 个外部位置、42 个去重外部名称；第 4 节 18+11+13 行覆盖全部 42 外部名称，9 个本地位置另列。当前 AIR 无候选位置，历史 AIR 18+8 名称另列。未修改旧 CSV、PLAN、论文、业务代码、结果、正式记忆或服务器文件。

路径均相对于项目根 `D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm`。SHA256 是本次只读获取，用于冻结所审版本，不表示重新验证论文数值或服务器产物。

| 来源 | SHA256 |
|---|---|
| `experiments/rgsp_table_completion_20261012_v1/CANDIDATE_PRIORITY.csv` | `cd354cccb0b43e1d9be332535b882029b71aedd8050ecfe30f4de4868dfc959d` |
| `experiments/rgsp_table_completion_20261012_v1/PLAN.md` | `a0f364773ba8967652bf6044d15592b4e0696eeb5963b3a3f906e6165fd10653` |
| `experiments/paper_table_fill_20261012_v1/input/main.tex` | `455929dbf0a81edb6195138b9562d6cf681816b3d0fe5768d496075a16546e98` |
| `reports/SSDD_COMPLEXITY_20261011.md` | `a71dae5e332f67f42ff8c7edab54b60ec6d5343caa7dca503a1632d6ac358726` |
| `reports/detector_complexity_evidence_20260724.md` | `8b56bd21e37e3850776e43d5385b387d98a33e6e75515d851919d26601f528d9` |
| `reports/four_dataset_methods_metrics_opensource_audit_20260823.md` | `ffec23f922ffccd95a3a5c4597e9de56881215c9d9343b5a9044d791b39637a6` |
| `experiments/rgsp_table_completion_20261012_v1/training/train_ssdd_v2.py` | `89543c9855575c59a707ea1d4d119ed3e367a6b0ae4ab87c29f2cb66aabf8f63` |
| `experiments/rgsp_table_completion_20261012_v1/training/frozen_config_source.py` | `ff1f6708e735d5bd91d9c341edfcfdcc714829c5ab5ee0c6110df263d5b6fed7` |

其他直接证据：`tools/create_ssdd_baseline_and_paper_doc.py:31`/`:34` 记录 ConsistencyDet/DiffusionDet 作者 URL；`reports/ssdd_comparison_reproducibility_audit_20260823.md:57` 记录 YOLOv10 官方 URL；主稿 `main.tex:1044`、`:1229` 起分别给 SADD/Ship 的实际转引协议；第 3 节各 profile 源文件及对应原始 JSON 保留在原目录。未为本报告创建第二份源数据或第二套当前状态。
