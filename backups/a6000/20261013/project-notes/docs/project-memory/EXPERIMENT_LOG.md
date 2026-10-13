# Experiment Log

说明：本文件只记录当前可访问材料中有数值或明确运行证据的实验。历史助手“已完成”的说法若没有结果文件或工具输出，会标为历史声称/部分可复现；计划脚本不写成已完成。不同 split、seed、初始化、指标和 batch 不合并成一个结果。

## EXP-OGSOD-MATCHED-01：OGSOD-1.0 单次匹配 TEST

- **日期**：日期未知；结果文件名/报告为 2026-07-31 附近
- **状态**：已完成，单次实验；三 seed 未完成
- **数据集/划分**：OGSOD-1.0，train/val/test images `14,664/1,834/1,833`，converted test boxes `4,769`
- **协议**：1024 输入、300 epoch、global BS64、双 H200、AdamW、FP32；验证集选择 `best.pt`，TEST 单次评估
- **Baseline**：Mamba-YOLO-T，P/R/AP50/AP75/AP50:95 `0.801434/0.711545/0.771733/0.402109/0.423018`
- **Ours**：IG-DS-SSM/PFIM-T，P/R/AP50/AP75/AP50:95 `0.797140/0.716924/0.772630/0.409711/0.436021`
- **结果**：AP50 `+0.090 pp`、AP75 `+0.760 pp`、AP50:95 `+1.300 pp`
- **复杂度**：约 5.9846M/5.9638M 参数；约 13.6/13.2 GFLOPs（640 reference）
- **限制**：只有一个 run；不能宣称稳定提升或普遍 AP50 超越。
- **证据**：`README_CURRENT.md`、`reports/aaai_paper_current_status.md`、`database/ogsod_test_results_20260731.csv`
- **可复现性**：部分可复现；后续已准备 `ogsod_t_primary_3seed_v2`，但本迁移未执行。

## EXP-SSDD-EX27：历史 PFIM ex27

- **日期**：2026-06-03 附近，精确运行时间未确认
- **状态**：历史恢复结果，已记录但协议不完全受控
- **数据集**：SSDD；具体历史 split 以服务器归档为准
- **模型**：Mamba-YOLO-T + DS-SSM/PFIM
- **结果**：P `0.97378`，R `0.95219`，AP50 `0.98704`，AP50:95 `0.74022`
- **复杂度**：约 5.9648M 参数、13.2079 GFLOPs
- **初始化**：历史脚本执行过 `model.load("yolov8n.pt")`；仅匹配约 6/519 项，实际大多数权重仍是随机/未迁移。不能重标为 clean scratch。
- **限制**：历史 baseline 与 ex27 的优化器、batch、学习率、warmup、代码状态和初始化入口不完全一致。
- **证据**：`README_CURRENT.md`、`reports/thread_019f2406_integrated_context.md`、`database/experiment_ranking_recomputed.csv`
- **可复现性**：部分可复现。

## EXP-SSDD-H100-SWEEP：SSDD 开发 sweep

- **状态**：历史摘要记录 `9/16` 已完成；该状态是开发阶段快照，不能替代最终验证。
- **协议**：742 train / 186 val / 232 official test；单 H100；640；global BS32；AdamW lr `2e-4`；300 epoch；改进模型曾使用 20 epoch module warmup；TTA 在已完成运行中低于 standard inference。
- **已知 VAL AP50:95**：baseline `0.548384`；P2-d16 `0.566725`；P3-d8 `0.561184`；P3-d32 `0.560158`；P4-d8 `0.558172`。
- **结论**：P2-d16 在当时暂时最好，提升主要来自 AP75；剩余配置完成前不能锁定 placement/state_dim。
- **证据**：`reports/thread_019f2406_integrated_context.md`。
- **可复现性**：部分可复现；当前状态需重新读服务器/日志确认。

## EXP-SAR-AIRCRAFT-PATCH：SAR-AIRcraft patch 协议

- **日期**：2026-07-21 至 2026-07-23
- **状态**：已完成的历史结果；非原图协议
- **数据**：512x512 overlapping patches；13,976 train、1,923 val、2,989 test；无 filename overlap
- **协议**：1024 输入、300 epoch、global BS48、FP32、AdamW lr `3e-4`、cosine、无 module warmup、无外部权重；增强为 mosaic 0.6、close mosaic 20、rotation 10、translation 0.1、scale 0.4、flipud/fliplr 0.5
- **Ours VAL**：P/R/AP50/AP50:95 `0.909612/0.796257/0.888252/0.705129`
- **Ours TEST**：P/R/AP50/AP50:95 `0.792764/0.637326/0.708220/0.555516`
- **历史 baseline**：VAL AP50/AP50:95 `0.877509/0.699035`；TEST `0.704018/0.545131`
- **复杂度**：5,964,583 参数、13.2 GFLOPs
- **限制**：baseline 使用 SGD、batch16、约 137 epoch 检查状态，和 ours 不匹配；外部 MaDiNet 原图协议不能直接排名。
- **证据**：`reports/sar_aircraft_experiment_20260721.md`。
- **可复现性**：部分可复现。

## EXP-SAR-SHIPDATASET：SAR-ShipDataset 历史对照

- **日期**：2026-07-23
- **状态**：已完成评估，但 baseline 对照不严格
- **数据**：31,783 train、3,972 val、3,974 test；输入 512；总计 39,729 images
- **Ours T TEST**：P/R/F1/AP50/AP50:95 `0.907854/0.889908/0.898790/0.948440/0.578182`
- **Ours S TEST**：`0.922367/0.915870/0.919107/0.956093/0.531767`
- **历史 Mamba-YOLO T**：`0.914006/0.890045/0.901866/0.946158/0.495684`
- **限制**：T baseline 训练中 batch 从 256 改为 128；B/S 参考不是同一完整训练协议；split 与 MaDiNet unpublished split 未确认等价。
- **证据**：`reports/sar_shipdataset_current_results.md`。
- **可复现性**：部分可复现。

## EXP-SADD-SCRATCH：SADD scratch

- **状态**：历史统一表/本地结果记录
- **协议**：scratch、200 epoch；P3-d16
- **Baseline**：AP50/AP75/AP50:95 `98.335/57.030/55.018`
- **Ours**：`97.760/56.985/54.597`，AP50:95 低 `0.421 pp`
- **另有记录**：P3 partial-transfer、300 epoch `98.297/82.999/72.113`，但缺少匹配 baseline，不能用作因果提升。
- **证据**：`reports/all_results_recomputed_20260831.md`、当前结果笔记。
- **可复现性**：部分可复现。

## EXP-FUSAR-PAPER-ALIGNED：FUSAR-Map 分割

- **状态**：已完成强 baseline/P3 诊断批次
- **协议**：四类 `[building, vegetation/other, water, road]`；WuHan_03_02 official test scene；17,560 train、4,390 val、1,280 test；256x256、batch10、Adam lr `1e-3`、每两 epoch 乘 `0.93`、150 epoch、FP32；weighted CE + Dice；验证 mIoU 选模。
- **模型**：Mamba-YOLO-T-style full backbone + UPerNet-lite；P3 位置被纠正为真实 P3/8。
- **Baseline**：best epoch 146；test mIoU/FWIoU/OA `44.2973/65.3817/77.1050%`；3.783M、2.366 GFLOPs、183.43 FPS。
- **Ours P3-d16**：best epoch 118；`43.5607/65.0833/76.7702%`；3.718M、2.329 GFLOPs、183.09 FPS。
- **Per-class test IoU**：baseline `[14.2797,71.4927,86.6344,4.7823]%`；ours `[15.2311,71.0746,86.8462,1.0908]%`。
- **观察**：P3 略改善 building/water，但 road 大幅下降；不能声称分割收益。旧五类/不兼容结果 mIoU 约 21% 不得与该四类结果混排。
- **证据**：当前结果笔记、`reports/all_results_recomputed_20260831.md`、`reports/design_evidence_matrix.md`。
- **可复现性**：部分可复现；迁移未重新运行。

## EXP-OPENEARTHMAP-BS64：OpenEarthMap-SAR

- **状态**：4/4 已完成
- **协议**：SAR-only P；VMamba-Tiny ImageNet backbone；P3-d16；输入 512，测试评估 1024；seed0；物理 BS32 + grad accumulation2，等效 global BS64；150 epoch。baseline 与 ours 使用同一 backbone 机会，新增模块随机初始化。
- **Baseline**：best epoch118；mIoU/FWIoU/mPA/Dice/OA `36.551/36.120/58.090/50.811/52.316%`；30.949M、42.321 GFLOPs、28.33 FPS。
- **P3 no warmup**：`35.898/35.220/54.914/50.163/51.764%`；31.275M、47.616G、25.41 FPS。
- **P3 warmup20**：`35.829/36.447/49.055/49.409/52.499%`；25.96 FPS。
- **P3 warmup50**：`35.535/36.104/49.563/49.099/52.383%`；25.93 FPS。
- **结论**：P3 未超过 baseline 的 mIoU；warmup20 的 FWIoU/OA略高但不改变 mIoU 结论。
- **证据**：当前结果笔记、A/B/F 服务器结果路径记录。
- **可复现性**：部分可复现。

## EXP-SAR-ACD-CORRECTED：SAR-ACD 分类校正批次

- **状态**：4 组已完成；此前旧版分类头结果不能混合
- **协议**：official VMamba-Tiny backbone 初始化；新增模块随机初始化；224；AdamW；200 epoch；seed0；BS64；只报告 Accuracy/Macro-F1。
- **Baseline**：Accuracy/Macro-F1 `89.764/89.585%`；29.483M、0.551G、119.99 FPS。
- **P3 no warmup**：`91.339/91.275%`；29.809M、0.804G、108.01 FPS。
- **P3 warmup20**：`96.457/96.410%`；110.17 FPS。
- **P3 warmup50**：`96.850/96.830%`；110.20 FPS。
- **代码问题与修复证据**：分类头此前未连接 P3 特征，导致预热阶段无梯度；校正后改为读取 `features[1]`。该修复由历史交接摘要记录，本迁移没有重新测试源码。
- **限制**：外部 MDB-DPL Accuracy `98.85` 使用不同六类协议，不能直接严格排名。

## EXP-CHANGE-DETECTION-BS：Ottawa / Yellow River / Farmland

- **状态**：BS16、32、64、96 批次均有记录为完成；不同 BS 是独立实验，不能合并或挑选。
- **统一条件**：150 epoch、scratch、P3-d16、imgsz192、AdamW、cosine、seed0；validation best checkpoint 在 TEST 评估。

| Dataset | Method | BS | P | R | F1 | IoU | OA |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ottawa | baseline | 16 | 90.259 | 94.000 | 92.091 | 85.342 | 98.351 |
| Ottawa | P3 | 16 | 88.777 | 96.003 | 92.249 | 85.612 | 98.352 |
| Ottawa | baseline | 96 | 87.953 | 95.147 | 91.409 | 84.177 | 98.173 |
| Ottawa | P3 | 96 | 89.841 | 94.346 | 92.039 | 85.251 | 98.333 |
| Yellow River | baseline | 16 | 91.205 | 85.702 | 88.368 | 79.160 | 97.817 |
| Yellow River | P3 | 16 | 91.422 | 82.787 | 86.891 | 76.820 | 97.584 |
| Yellow River | baseline | 96 | 89.377 | 78.320 | 83.484 | 71.650 | 97.002 |
| Yellow River | P3 | 96 | 91.218 | 71.146 | 79.941 | 66.585 | 96.546 |
| Farmland | baseline | 16 | 77.704 | 13.887 | 23.564 | 13.355 | 92.778 |
| Farmland | P3 | 16 | 93.259 | 49.213 | 64.428 | 47.523 | 95.644 |
| Farmland | baseline | 96 | 80.623 | 12.532 | 21.692 | 12.165 | 92.747 |
| Farmland | P3 | 96 | 91.747 | 50.719 | 65.325 | 48.506 | 95.684 |

- **结论**：P3 在 Farmland 显著改善召回/F1；Ottawa 小幅正向；Yellow River 负向。
- **证据**：当前结果笔记、`reports/current_cross_task_results_20260904.md`（该文件较早，部分 queued 状态已被后续结果替代）。
- **可复现性**：部分可复现。

## EXP-COMPLEXITY-MADINET：复杂度实测

- **状态**：部分完成
- **环境**：3090、PyTorch 2.3.0+cu121、FP32、batch1、输入640
- **结果**：Mamba-YOLO-T 5.9855M/13.6116G/26.3284ms/37.9817FPS；DS-SSM 5.9648M/13.2079G/28.7546ms/34.777FPS；DG-DS-SSM 5.9632M/13.1554G/29.0699ms/34.3998FPS。
- **MaDiNet**：只测得 official MambaSAR res4+res5 核心组件 223.2648M、345.283G、27.2921ms；不是完整 detector，不能冒充端到端复杂度。
- **证据**：`reports/madinet_ds_ssm_complexity_report.md`。

## EXP-SSDD-SCRATCH-B16：SSDD 当前受控 scratch（batch16）

- **日期**：2026-08 下旬跨任务矩阵期间（A6000）
- **状态**：已完成（seed0）
- **协议**：SSDD 742 train / 186 val / 232 official test；150 epoch；batch16；AdamW lr `2e-4`；scratch `weights=none`；sd16/sd32；无 module warmup
- **Baseline（d16 结构）**：AP50/AP75/AP50:95 `0.834389/0.193933/0.352387`
- **P3-d16**：`0.881374/0.337564/0.429913`（AP50:95 `+7.752 pp`）
- **P3-d32**：`0.862660/0.286460/0.405960`
- **结论**：当前受控证据支持 d16 优于 d32。
- **证据**：`reports/thread_019f2406_integrated_context.md`、压缩线程 `019f2406` 交接摘要；结果文件在 A6000 `/home/powerleader/ig-ds-ssm/experiments/sar4/` 与 `ssdd_bs32/`

## EXP-SSDD-SCRATCH-B32：SSDD BS32（A6000-F）

- **状态**：已完成（seed0）
- **协议**：同上但 global BS32
- **Baseline**：AP50/AP75/AP50:95 `88.629/42.497/45.683%`
- **P3-d16**：`88.484/32.438/42.486%`
- **结论**：该 BS32 运行中 P3 落后；不能声称 BS32 普遍更好，BS16 与 BS32 结果必须分开标注。
- **证据**：压缩线程 `019f2406` 交接摘要；`eval_test.json` 在 A6000-F `experiments/ssdd_bs32/`
- **补充**：SSDD/SADD 的 seed1/2 BS32 队列（A6000-A/B/C/D，2026-08-31 启动）记录为已完成且 `eval_test.json` 存在，但具体指标尚未汇总进任何表格；**待核实**。

## EXP-SSDD-HIST-RECHECK：历史 checkpoint 固定官方 TEST 复核

- **日期**：2026-08-28
- **状态**：已完成
- **协议**：历史恢复权重在统一 232 图官方 TEST、FP32、640、batch32 下复评
- **历史 Mamba-YOLO-T baseline**：P/R/AP50/AP75/AP50:95 `0.967385/0.961538/0.986050/0.836129/0.703479`
- **历史 PFIM ex27**：`0.977586/0.958566/0.986881/0.840055/0.714454`（ex27 对历史 baseline AP50:95 `+1.098 pp`）
- **关键结论**：ex27 的 `0.74022` 是历史验证集数值；官方 TEST 复核为 `0.714454`。当前 scratch 与历史结果的绝对差距主要来自训练集 928 vs 742、300 vs 150 epoch、超参与旧拓扑，而非 `yolov8n.pt`（仅匹配约 6/519 张量）。
- **权重兼容性**：历史 baseline→当前 baseline 528/528 张量；ex27→当前 P3-d16 450/519（约 86.7%）。ex27 只可作 SSDD 目标域 warm-start 诊断，禁止标为外部预训练。
- **证据**：`reports/ssdd_historical_recheck_and_fusar_audit_20260828.md`、`artifacts/ssdd_historical_checkpoint_diagnostic_20260828/`
- **补充（历史验证集排名）**：PFIM ex24 AP50:95 `0.73571`、DS-SSM ex7 `0.73520`、DDP 复现 `ex27-codex3` `0.73028`。
- **待核实**：warm-start 配对诊断脚本启动时返回 `ALREADY_RUNNING`，可能是 pgrep 误匹配；是否真正运行及结果未确认。

## EXP-SRSDD-HBB-V1：SRSDD-v1.0 旧 HBB 管线（协议存疑）

- **状态**：已完成但标记为协议存疑/诊断
- **数据**：666 图、2884 OBB、6 类（Container 2053、Fishing 288、Dredger 263、ore-oil 166、Cell-Container 89、LawEnforce 25）；92 条越界/负坐标标注
- **结果（seed0 batch16）**：baseline AP50/AP75/AP50:95 `0.188064/0.028232/0.060870`；P3-d16 `0.184199/0.020640/0.056944`
- **限制**：原生 OBB 被当 HBB 处理，协议与发表值不可比；v2 已改为 177 条裁剪标注的显式 HBB 转换（426/106/134）并保留官方 OBB 证据。
- **外部参照（作者报告）**：GDB-YOLOv5s（IET Image Processing 2024，DOI `10.1049/ipr2.13140`）SRSDD HBB：YOLOv5s AP50 `80.2`/AP50:95 `42.4`；GDB-YOLOv5s `81.7`/`47.6`。

## EXP-AIRSHIP-TILES：AIR-SARShip-2.0 tiling 受控 scratch

- **状态**：已完成（seed0，matched）
- **数据**：300 张 1000x1000 TIFF、2040 框；旧整图缩放到 640 的结果（AP50≈0.01）判为无效作废
- **协议**：源图级划分后 640 tile；scratch；matched baseline/P3
- **结果**：baseline AP50/AP75/AP50:95 `47.590/7.463/17.375%`；P3 `51.559/9.087/19.291%`
- **外部协议差异**：Bao et al. 2021（互补预训练）使用 210 train+val/90 test、预训练 SSD 系列，AP50:95 46-55 区间；scratch Mamba-YOLO 不能直接比。已建 paper-aligned 轨道（512 裁剪 256 重叠、1890/810 patch、YOLO 配方 416/bs12/SGD 240e、fixed-last）。

## EXP-SARMIX-PROXY：四源混合数据代理消融

- **状态**：部署运行过，最终汇总状态**待核实**（本地存在 `output/sarmix_scratch_d16_v1/` 与 `sarmix_scratch_d16_b8mem_v1/` 摘要）
- **设计**：SSDD 200/80 + OGSOD 250/100 + SAR-Ship 200/80 + SAR-Aircraft 350/140 = 1000 train/400 val、11 类；scratch、sd16、512、bs16、150 epoch、seed0
- **配置**：baseline、full P2/P3/P4/P5、P2+P3、DS-only P2、PFIM-only P2、global P2 共 9 个；分片部署在 lab540-3090-5 与 3090-1
- **证据**：压缩线程 `019f2406` 交接摘要；`reports/duplicate_report.md` 等未直接引用；本地输出摘要文件存在

## EXP-SHARED-V1-INVALID：旧共享任务迷你 runner 结果（作废标记）

- **状态**：已完成但判定不可靠，不得作为论文证据
- **内容**：SAR-ACD baseline Acc `99.21875%` / ours `98.4375%`（约 1.98M/1.77M 参数迷你 VSS backbone）；SAMPLE 双双 `100%`（简易随机视角划分）；Ottawa/Farmland F1=0、Sulzberger 异常（abs-diff 输入、仅 2-3 训练 patch、逐 batch 平均）
- **处置**：由 v2（`prepare_cross_task_v2.py` + `run_shared_sar_tasks_v2.py`）取代；v1 数值仅保留作历史诊断。

## EXP-SSDD-3SEED-FAIR：SSDD 检测 3 seed 公平消融（主结果）

- **日期**：2026-10-01 至 2026-10-02
- **状态**：已完成；15 个 run（5 变体 × 3 seed）全部跑完并汇总
- **数据集/划分**：SSDD，928 图 → 742 train / 186 val（`random.Random(seed).shuffle` 确定性切分）/ 232 official test 封存
- **协议**：scratch（weights=none）、batch=16、imgsz=640、epochs=150、patience=30、AdamW lr0=0.0002 lrf=0.01、weight_decay=0.0005、mosaic=0.3 close_mosaic=15、deterministic、seed∈{0,1,2}；同 seed 同 split（paired），验证集选 best.pt，官方 test 评估
- **变体语义**：baseline（VSSBlock）、full_p3（DS_VSSBlockSweep 默认，DS-SSM+PFIM）、ds_only_p3（NoPFIM，仅 DS-SSM）、pfim_only_p3（仅 PFIM，λ_s=0/α=0）、global_p3（PFIM image-wise）
- **执行**：powerleader-a6000-f（RTX A6000×2）；runner `run_sar4_full_suite.py`
- **AP50:95（mean±std，3 seed）**：
  - baseline `0.4203±0.0347`（0.4269/0.3827/0.4512）
  - full_p3 `0.4291±0.0113`，Δ +0.88pp（−0.35/+5.93/−2.94）
  - ds_only_p3 `0.4398±0.0144`，Δ **+1.95pp**（+2.90/+5.26/−2.30）
  - pfim_only_p3 `0.4312±0.0170`，Δ +1.09pp（+2.13/+3.16/−2.00）
  - global_p3 `0.4377±0.0165`，Δ +1.74pp（+2.29/+6.15/−3.23）
- **科学结论**：
  1. Q1（DS-SSM 单独）= +1.95pp，均值最高、最稳健核心增益，AP75 +3.53pp。
  2. Q2（PFIM 单独）= +1.09pp，召回提升最明显（R 0.8139 vs 0.7924）。
  3. Q2'（PFIM on DS-SSM 边际 = full − ds_only）= −1.07pp，朴素结合有负交互；full_p3 三 seed 2/3 为负。
  4. Q3（token vs image）= global 优于 full（+0.86pp），token-wise 无稳定优势。
  5. seed2 baseline 为 outlier（0.4512），导致 seed2 上 4 变体全负；SSDD 单 seed 方差大（baseline AP50:95 std 0.0347），须多 seed 报告方差。
- **复杂度（seed0 实测）**：baseline 5,985,547/13.61GFLOPs/6.62ms；ds_only 5,918,104/13.12/6.00；full 5,920,441/13.15/6.09。轻量化部分来自架构替换（DeSpeckleSSM state_dim=16 + MLP mlp_ratio 2.0），非纯 SSM 机制增益。
- **对论文方向的影响**：原"token-wise 优于 image-wise / DS-SSM+PFIM 联合稳定提升"论断不被 3 seed 证据支持；建议主打 DS-SSM 单独稳健增益 + PFIM 独立召回提升，诚实报告负交互与 token-wise 无优势。
- **独立核验（2026-10-02，批次 round96-kimi-k3）**：只读核对本地结果、项目记忆与 A6000-F 服务器产物：15/15 个 run 均有测试评估与权重；本地 `results_individual.csv` 的 15 行指标与服务器评估逐项吻合到 6 位小数，权重 SHA256 一致；服务器冻结记录的代码哈希与当前本地、远端代码一致；文件名层面未发现 train/val 与官方 test 重叠。
- **结论边界补充（2026-10-02 核验，批次 round96-kimi-k3）**：ds_only 与 pfim_only 均只有 2/3 seed 配对差值为正，AP50:95 均值领先不足以支持稳定/普遍提升结论；三个 seed 的 train/val 文件哈希彼此不同（同 seed 内变体共享 split），跨 seed 波动同时包含训练随机性与 split 差异，"seed2 baseline 高是 split 更容易"目前仅为猜测；DS 变体约 5,918,104 参数、较 baseline 少约 1.1%，精度/速度差异不能全部归因于 DS-SSM 机制，若要加强因果结论需补同结构控制。
- **证据**：`dsh_experiment_results/RESULTS.md`、`results.csv`、`results_individual.csv`（含权重 sha256）、`BEST_CONFIG.md`、`PAPER_EXPERIMENT_SUMMARY.md`；远端 `~/ig-ds-ssm/experiments/ssdd_paper_v1/seed*_ssdd_<variant>/eval_test.json`
- **可复现性**：可复现（协议冻结、clean scratch、每 seed 确定性划分、权重 sha256 记录）

## EXP-RGSP-TABLE-COMPLETION-20261012：快速补表与新SSDD四臂seed0

- ProjectKey：Mamba；FactKey：Mamba.Experiment.TableCompletion20261012；事件日期：2026-10-12（+08:00）；EvidenceStatus：CONFIRMED（已回收产物、来源哈希及静态核验）；LifecycleStatus：EXPERIMENTAL；RunId：`rgsp-table-completion-20261012-01a12dd6-7cb894b2`。
- 本轮用户授权组织快速实验补表；独立根 `experiments/rgsp_table_completion_20261012_v1/`，PLAN.md与下述原始回执是证据，不是第二套正式记忆。仅使用已登记A/B/C/D/F已有环境；先盘点并避开B旧AIR锁，不抢占/删除锁、不改旧服务与队列；新任务只获非阻塞锁后启动。
- 现行补表稿：`experiments/paper_table_fill_20261012_v1/input/main.tex` SHA `455929dbf0a81edb6195138b9562d6cf681816b3d0fe5768d496075a16546e98`。六张原主表填24个原空格（12 Params+12 covered FLOPs），290→266（152性能/114复杂度）；Ship新表21值（15性能+3 Params+3 covered FLOPs）全齐。总新增45值不等于填45原空格；POST_FILL_GAPS.json绑定同一主稿SHA。
- 数值/源码验收：三份 MAIN_*_STATIC_VALIDATION.json及DETECTION_REVIEW_ACCEPTED均PASS；各阶段逆补丁保留旧数值与无关正文，72唯一标签/12表。95份cross源清单SHA/大小、27个reuse候选值、38份Ship捕获字节文件及24项receipt源端哈希均匹配。此轮记忆核验未重取权重、重算全数据或重新跑模型。
- Ship已回收三臂seed0：31735/3970/3974、150epoch、resumed=false、640/FP32/BS16；Original/Matched075/Full AP50:95为51.904208/50.379574/51.169528%，Full−Original −0.734680pp、Full−Matched075 +0.789954pp。Params为5.985547/5.917907/5.920441M完整注册参数；runtime_audit单独捕获核SHA，不是8项receipt之一。共同manifest `2b30f37c7eaa9d5118354736959e64035841a3219ea3fd44963ae52ee6fbda78`。保留负结果，不冒充旧T/S/ex27或多seed稳定提升。
- 跨任务profile：A对SAR-ACD/SAMPLE/Ottawa各Original/Matched075/Full共9模型测量；分类单张224²、Ottawa双张192²，共享主干前向两次但参数一次。FP32/batch1；10份回收JSON匹配，聚合SHA `3d79452341cfc226fb696f51b19cb66b0e25ea2d099c9566faa7f9b57f3d846a`。结果/config/checkpoint/脚本已绑定，失败run1现场保留。
- 检测profile：A测单类640²完整dense detector的cnn_yolov8n/Original/Matched075/Full，Params为3.011043/5.985547/5.917907/5.920441M，covered FLOPs为8.194150/17.914822/16.865504/16.898477G。随机初始化、未加载训练权重；5份回收JSON匹配，聚合SHA `0aaad345c8ad22b4780ec11a6828695b07e3c2d1b948c4ce3917c1bd282abf75`。NUMERIC_REVIEW独立复算13结构PASS；只按配置绑定回填SADD当前seed0组6格及新Ship3格，不套到旧T/S/ex27、外部方法或不同nc/输入。
- 计数边界：旧THOP口径×2加einsum、因式化scan基本算术及CrossMerge，是覆盖算子估算；不计scan特殊函数、未覆盖功能式运算/插值/统计、部分激活及Ottawa GroupNorm、CUDA前缀额外操作/搬运、预处理/NMS。保留diamond及条件，不称精确总FLOPs、速度或与外部统一排名。
- 短测：D四模型×BS4积累8/BS16积累2共8项，均PROBE_PASS、heldout_labels_read=false、paper_eligible=false、12计时updates；30份小文件SHA核验PASS。只读928训练图/标签。Mamba BS16训练步仅提速约3–4%，故正式统一BS4×8保留显存余量；不同物理batch不宣称训练轨迹等价。吞吐详见training/BENCHMARK_AND_LAUNCH.md，不填论文精度。
- 两条完整链路smoke：D YOLOv8n v1、F1 Full v2均1epoch/29updates、SMOKE_PASS、paper_eligible=false；FP32保存→重载→最终验证通过。来源training/evidence/smoke_yolov8n_b4_v1_result.json与smoke_full_b4_v2/{result,runtime_audit}.json，Full独立JSON与F原始快照源SHA绑定。smoke读取TEST-as-VAL属于明确选模验证，不将其指标当正式训练结果。
- 正式冻结配方：SSDD 928 train/官方232 TEST-as-VAL；scratch/seed0/640/FP32，AMP/TF32关闭；物理BS4×8有效32，每epoch29次样本归一化更新、共300epoch/8700updates；AdamW1e-3、5epoch优化器预热+cosine至1e-5、无模块预热/早停；仅水平/垂直翻转0.5+letterbox，EMA按0.1AP50+0.9AP50:95选best且同分取更早。官方test反复选模，禁止称独立测试。
- 入口差异与核验：D继续v1（SHA `d01faeb54cd93935fe795d8d1cf84263e9c3b2c6212fcef950327148940b010e`）；三Mamba用v2（SHA `89543c9855575c59a707ea1d4d119ed3e367a6b0ae4ab87c29f2cb66aabf8f63`），修正dt_projs_bias等bias别名免WD并记录精确组名。D实际第51epoch最佳checkpoint经服务器CPU反序列化逐组核对v1/v2对YOLO完全等价，AUDIT_PASS；此等价性不外推Mamba，旧v1现场保留。
- 首批运行仅为分时快照：D 10:49:16核YOLOv8n 145epoch/4205updates；C 10:51:07核Original 3/87，F 10:51:06核F0 Matched075及F1 Full各3/87。四项均RUNNING_VERIFIED_NOT_COMPLETE，performance_results_ready=0；不是当前live或正式300epoch完成。FIRST_BATCH_STATUS.json SHA `de3c7c545e32551f3779b6f4c97af0d735a47b38191ce343f2080a788fad13f5` 汇总于10:52；其中3原始快照hash、实际PID/启动tick/GPU/启动回执/ledger已对照，后续不得按旧PID操作。
- 四臂冻结模型源与数据清单一致，三Mamba共同初始张量SHA一致；训练入口v1/v2不同不能写成源码完全同一。数据manifest SHA `8c8db821f87451251bc670b4b69f1a70a55ceeb1ffd8ba91f7f6bf70447eef05`；350项训练源manifest SHA `149133bdfefb969279040921719bfb79b5d307e0a8c350d9f8214867b465c502`；共同tensor SHA `1b6255d416fbf073c1ffc896e0976471c9a234c374955b76e857b2afed704303`。profile的360项清单另属其固定范围，不混报数量。
- 后续：CANDIDATE_PRIORITY.csv为110配置×2seed的220候选，4已启动/216未启动；不是全量就绪。SAMPLE806合成14–16°→539实测17°源齐，新128/BS36/SGD0.001恒定LR/10000updates/末权重协议正在适配准备，未正式训练；真实域仅最终评估。SADD2966含2083空标签源存在但全量协议/场景独立未验；关联SAR-ACD仅五类2522图、六类缺权威第六类来源；Ottawa旧GT监督不能当无监督完整实现。
- 可编辑源码交付：`delivery/RGSP_table_completion_20261012_source_only_v1.zip`，550成员、51,099,771bytes，SHA `f42c06663d9efaee94dc94db9b03101918500aefe783acc349fc34de554a1ced`。包级与独立STATIC_PASS_SOURCE_ONLY回执核548复制文件/440原输入、成员CRC/SHA及7图路径；包不含新训练目录/权重/可变PLAN。内建编译尝试因Noto Serif CJK SC缺失失败（input/$build/main.log），没有新论文PDF、分页或视觉验收；未安装字体。
- 旧57项seed0全验收且AIR三个阻塞位置解决前不开旧seed1，不缩54、不关闭旧研究矩阵、统一效率/机制或英文P2/4冲突。保留全部历史正负结果；本轮已交付源码和启动首批训练，不是完成全部研究实验。后续须独立回收真实完成记录、FP32最佳权重重载和统一指标，才可填正式成绩。
### 首个新协议正式结果续接（2026-10-12）

- 新增完成核验（11:20后续）：D YOLOv8n已300轮/8700更新，最佳第168轮；P/R/AP50/AP75/AP50:95=95.0005/89.0110/95.4469/80.9772/68.1836。300行连续账本/CSV、FP32重载、最终最佳权重现场SHA均通过；见独立任务training/YOLOV8N_COMPLETION_VERIFIED.json。该结果仅属928/232 TEST-as-VAL新协议，已生成独立TeX片段，尚未并入封存主稿。其余三臂仍待完成；不关闭任务。
- 最佳权重SHA256：263192d557642b47d8d316d06dfe8a7ce54819327b5645d0113861f135641d41；只在D核hash，未下载权重。原始现场快照为training/evidence/completed_yolov8n_20261012.json。YOLOv8s在候选清单属于SADD，未执行额外SSDD YOLOv8s。

## EXP-UNIFIED-A6000-20261012-V2

- EventDate：2026-10-12 +08:00；CONFIRMED/EXPERIMENTAL；用户授权续接并统一优化配方，证据根 `experiments/rgsp_unified_a6000_20261012_v2`。30独立配置已分配A/C/D/F，B旧任务保留。最多3/2/1seed依实测耗时而定，旧57项seed0门槛未改变。
- 21:47逐机证据：9完整seed0 result/completion/protocol/权重SHA核验通过（A5监督CD，C3 SAMPLE，F1 SSDD YOLOv8n）。不是同刻全机扫描，不据此宣称全部训练完成。归档SHA及运行ID见STATUS_20261012_2147.md。
- 新SSDD YOLOv8n：730/186/232独立test，127e/best97；P/R/AP50/AP75/AP50:95百分比89.8361/75.4579/83.2336/30.2989/39.1759。结果仅此统一150e新协议，不覆盖旧928/232 TEST-as-VAL300e成绩。
- D d0r2三模型训练后因skimage缺失评估失败、两模型pywt缺失短测失败；经用户明确授权task-local安装scikit-image0.24.0/PyWavelets1.6.0及小依赖，wheel哈希留install01.json。d0r3五方法新短测全部PASS并正式放行；GT仅选定权重推断后读取，图像伪标签transductive协议与A监督空间holdout不同，不能直接公平排名。
- SADD native全2966图/883阳/2083阴，2372/296/298；manifest SHA3440521e8920f8043e13fe43b4f31b5fb12310ed854497ee9756f086d5a81b09；原始框未修/裁。首次gate缺前置production state退出已保存，3gate回归测试通过后d0sadd2排在d0r3结束且PID退出之后。33旧测试+2新gate的D全套35PASS，之后3gate PASS。
- SAR-ACD3032图官方pinned包20跨类别RGB组/40图冲突；用户明确保留原包等待权威修正版，未训练或删除。官方包/审计由C保留；不以模型转述升级原生证据。
- 未完：剩余训练/种子、外部CD效率、110候选未验证实现/协议、四臂300e旧结果独立整合。RunId：`rgsp-unified-resume-20261012-01a1301f`；本条不关闭研究/论文TODO。

- 后续核验22:13：25次完整运行/18配置（A13/C7/D4/F1=18seed0+7seed1），7组同配置配对protocol SHA一致；D LANTNet/CAMixer/WBANet/WASNet已完整最终评估/outputs核验，早前skimage问题已在新尝试解决。四份2213证据包本地SHA全部匹配，见STATUS_20261012_2213.md；其余队列仍未完成。

- 旧300e补表核验：C Original/F Matched075+Full/D YOLOv8n全部完成300e/8700updates，300行ledger/CSV、实时best和data SHA通过，三个old300_*_2217证据包本地SHA一致；OLD_SSDD_300E_RESULTS.md/tex单独归并四臂P/R/AP50/AP75/AP50:95，Full−Original AP50:95=-0.5329pp保留。未重推断或重载，FP32重载为原生产者回执。新SAMPLE四个seed0 real17 accuracy均低于5%，类别映射核查一致，原因未确定，不改test/筛除结果。


## EXP-PERFORMANCE-FILL-20261013：新结果与对应复杂度补表

- 日期：2026-10-13 09:33 +08:00；用户授权补最新45dee9dc附件，保留正文与旧数据、增加新协议组。
- 已核时点：09:15 A/C/D/F生产计划/worker验证67次完整运行（27/12/19/9），错误为空；09:20回取成绩与09:15一致，结果字节SHA与completion匹配；26配置已有结果、22配置计划种子齐。B旧AIR等ready不接管；其余未完成不由本项关闭。
- 表格：新SSDD四臂共同seed0/1，均值±样本标准差；YOLOv8n额外seed2不混入。AP50:95 YOLO/Original/Matched/Full =38.64±0.76/41.83±2.67/43.00±0.57/42.21±0.69，保持负结果。SADD全2372/296/298（含2083阴性）四seed0 AP50:95 n/s/Matched/Full=54.20/57.06/53.60/52.88；Original继续，旧795/44/44单独保留。新Ship仅配置复杂度，成绩未齐。
- 本项新增45性能值；56个原空复杂度格+24新增组复杂度格（80格）。旧SAR-ACD五类/SAMPLE real-only/Ottawa冻结检查点复杂度复用已审计结果；新运行直接读取各自PASS覆盖profile。未绑定确切配置的文献/历史空项仍保留，不以默认规模代填。
- A6000 A为idle目标，锁当前/旧campaign GPU，60s无compute，单批1FP32补测DDNet/LANTNet/CAMixer/WBANet/WASNet。参数=6712/24956/34214/7159/7360，与正式生产一致；FLOPsG=0.000235200/0.002287840/0.000577862/0.000045180/0.000052608，原生3×7/7/9/8/8图块。模型/registry/覆盖profile源SHA本地远端相同；未载训练权重、无新训练。复用D已授权的任务内PyWavelets1.6包到A本次独立deps，共享环境未改。
- 覆盖限制：MAC计2，Mamba含THOP+扫描基本算术/einsum/CrossMerge；未穷尽指数/softplus/GELU/SiLU/函数式统计门控/插值/解码/NMS，监督CD另缺GroupNorm。外部CD仍未穷尽OpenCV DCT/函数式小波卷积/注意力matmul/激活统计，单中心图块不是整图累计量；没有宣称统一全算子FLOPs或速度优势。
- 交付：`paper_draft/rgsp_performance_data_fill_20261013/performance_comparison_filled.tex` SHA `814b93624e5f7f1fbbc1ef4d28333b9921e5c0aada3e505d592d1c142360c2ea`；源附件SHA `091f2bf201df7a464213e621cb83dcfab91a13853d019b9f91bcbf0b6fe4d2b5`。完整指标/协议/编辑操作/静态验收均在其evidence与DATA_FILL_EVIDENCE。保留六表/全部表外正文/旧性能数值，逆向原稿恢复、数学分隔符、括号/环境/唯一标签均PASS；未整稿编译或分页验收。
- 原生外部profile `experiments/rgsp_profile_fill_20261013_v1/results.remote.json` 与远端结果SHA相同 `a7d21753bf9d35bdbdb2a08a41caf8ff78bf339e17ef73a789ba044b1a04f691`；本地results.json为增加末换行的同内容便读副本，SHA不同不冒充原件。PyWavelets运行包SHA `cf6723aac568b337079861dfc46390f0fb9bb38d17d26fe6beb6be400472df54`。
- 状态：本次补表交付完成；当前实验、科研公平性/机制/效率验收与全部110候选仍未完成。RunId `rgsp-performance-fill-20261013-01a1301f`。

## EXP-RGSP-REBALANCE-20261013 · 续接重分配验收

- ProjectKey=Mamba；FactKey=Mamba.Experiment.UnifiedA6000V2；EvidenceStatus=CONFIRMED；LifecycleStatus=EXPERIMENTAL；事件2026-10-13 +08:00；记录时间2026-10-13T10:16:27+08:00；RunId=rgsp-rebalance-resume-20261013-1003。
- 当前用户授权：分析两段指定聊天并补全前者。前者最后中断于A补SAMPLE与C接手Ship Full后，既有训练服务器后台仍运行。本轮读取共享入口/正式记忆/两段聊天相关用户及结果，核对登记SSH目标A/B/C/D/F，没有从旧历史获取新账户，也未向其他聊天发送消息。
- 10:03资源盘点五台A6000均可达，磁盘A195/B587/C764/D94/F197GB，CPU负载正常；A短任务、C长任务合理，D/F既有任务继续，B旧AIR持锁不接管。A六基线预检与生产探测通过；10:10已见AlexNet真实50轮，最终83轮/913更新早停、最佳53轮。C Full首轮1164.13秒；F Original/Matched075和D SADD账本持续。原30配置加6补充共36个配置，Ship转移不重复计数。
- 分时产物回收：首批A27/C12/D19/F9=67次；A后续28替代旧A计数后合计68次/27配置，27seed0/22seed1/19seed2；704小文件SHA、协议/完成回执、checkpoint现场哈希PASS。原30配置22组种子齐，不是36配置全完成。原67次及此前25次证据保留。
- C生产Ship Full与F原probe协议完全一致，SHA caed014dc9803c4aa888caa8330ee4d7b5528aa0f95ed5a2990c0a1b40e2ed82；原聊天的SIGTERM只作用调度worker，冻结处理器等当前Matched075子进程完成后退出。F1 Full PENDING属于保留历史；C才是现行owner。启动器保留attempt且不自动故障恢复；禁止按旧PENDING再次派发。
- SAMPLE六补充基线与C既有SAMPLE公共协议（除方法名）一致，645/161/539，2690源文件复制核验回执保留；AlexNet、GoogLeNet、VGG16无BN、ViT-B/16、Swin-T/B均scratch。单seed预估4.6–23分钟，每项最多3seed，实际seed0耗时可降低后续上限。新增的是通用补充对照，不能记作原110候选的专用SAMPLE方法复现。
- AlexNet seed0独立test：Accuracy9.4619666%、Macro-F1 3.5051546%；57.044810M参数、1.420303744G covered FLOPs，权重SHA b2c37685af83b4a556c1b81706ec0becb1fcef6dd2b10a492d6214397e293b7e。最佳合成val Macro-F1 73.1268%；实测域负结果保留，单seed不报std，不据test调参。输入224²/batch1测复杂度，覆盖非穷尽。SUPPLEMENTAL_RESULTS独立登记，旧论文未被改写。
- 产物：experiments/rgsp_rebalance_20261013_v1/{README.md,Read-LiveStatus.ps1,read_live_status.py,verify_resume_evidence.py,SUPPLEMENTAL_RESULTS.md}；evidence/{RESUME_1006_VERIFICATION.json,RESUME_1014_VERIFICATION.json,resume_live_final.json,c_first_epoch_1014.json}。统一campaign README加当前分配入口。只读SSH脚本已对五机实际执行；静态JSON/哈希审计在本机处理文件，无torch/模型导入，没有新增本机运行实验。
- 尚未完成：SADD余种子、三项Ship、六补充基线剩余种子；SAR-ACD权威修正版、AIR冲突及源码/准确配置缺口继续阻塞。C按首轮剩149轮外推约48.2h，整体暂估10月15日上午，非承诺。重分配交接完成不等于研究验收完成。


## EXP-SARACD-PAIRED-20261013 · 六类两版及方法优先队列

- CONFIRMED/EXPERIMENTAL；用户在本聊天明确授权3032与2992都训练测试、释放B旧AIR空等组、保持BS16×累积4有效64/官方明确结构、优先Original/Matched075/Full；最后要求排队后不持续监测，等待其通知完成。此前等权威修正版决定被本授权替代；原文件、旧57门槛/AIR3位置不变。
- C按指定git clone官方仓库，commit537639c406d5d37e52ba75fa433de52db7d1d483，3032图全解码、六类；20组跨类同RGB共40图。B无git，不安装；导出含.git、README及3033 tracked_file哈希，B/A复核完整来源。旧五类2522缺A320/321 510张，不再误报为完整六类。
- 新manifest：full1000/240/1792，clean960/240/1792；每类干净train160/val40，冲突40仅full/train。共同holdout SHA51da1dae01bcf11288d85d57514d347867581a9ee7959023a46eda94e0930400。full manifest917e638f6bb9e8111e7d83d1b8fcc45dade396e1ac66c40645415e349c1dfc49，clean24fe4b0f7877bbaf11cda98c4d7c31a072fa8edda2ebb68c1eb9bcbac174c610。四项错误清单拒绝fixture通过；源图未删改，非官方划分/非场景独立，150轮带来16/15updates每轮差异。
- B释放：10:27:13回执RELEASED，已确认旧三等待进程身份/无GPU计算和checkpoint后仅结束该等待组，所有原文件及恢复记录保留；不按旧PID恢复。部署首次source archive未传完导致SHA断言失败，runtime已解/source空；随后完整hash通过后只解source，无覆盖旧实验或软件安装。
- 新根rgsp_table_expansion_20261013_v1；B六核心probe PASS（单次150e外推13.1–15.4分钟），核心stage0、ResNet50stage20；调度仅stage改变，旧状态/计划/回执有快照，当前planSHA2c1a3f1c5213ca158ff0ce0954ccf1af8b0aba94db3c83d8bcd4bb825dc436bd。首正式clean2992 Matched075 seed0 epoch1=5.894997747秒、960图/60microbatch/15updates；14分钟不是每epoch。B20秒GPU样本21–82%/2452MiB，未实测同卡并发收益或更改BS。
- A保留原SAMPLE训练；12个外部分类配置接在a0samplex终态且PID退出后，B核心及ResNet50共8配置，合计20配置，最多3/2/1seed依耗时。具体AlexNet/GoogLeNet(aux off)/VGG16无BN/ResNet50 v1.5/ViT-B16/Swin-T/B以adapter身份为准，不能映射为未明确规模的文献泛名称。
- 补齐缺项工作：新增TorchVision0.14.1 R50-FPN FasterRCNN/RetinaNet/FCOS runner，SSDD730/186/232；原生loss、batch16累积4、FP32、best-val EMA后test、P/R/AP50/AP75/AP50:95、Params及部分覆盖FLOPs/延迟。B已通过框/标签转换、空标签/预测、重复匹配、三个官方构造器、730train源哈希fixture；GPU6update+保存重载预检后置b0det，当前GPU阶段未执行，不声称复现性能完成。尾部profile源静态PASS，SHA79e90ddd4cede416dc6f40549f9739ec8002b9064c22c56cad1c24fe93cac4e4。
- 10:53交接快照：B92/A12共104小文件SHA PASS，归档SHA c2aab2ceb5b6756386978454174278516c7ed4b47cda6eaff4af77c787725204 / d6f633b728aaba0275b932c0aa35cecbbb35655cdcb67ab2d0e2dc0c6ae6aa61。C原生clone回执另回收。该快照是移交证据，不是最终训练验收；结果与权重留服务器。
- 时间估计：核心6seed0约1.5–2h，最多18核心run约4.5–6h；全20分类配置最多计划seed约14–18h，A外部用旧SAMPLE吞吐按训练量缩放并含原队列余量；早停可缩短，不含GPU预检未执行的三检测或专用模型源码缺口。
- 入口README及evidence/HANDOFF_VERIFIED.json；未改论文既有数值/共享环境/服务，未本机模型训练推理，未新增监测automation。用户通知后再核完成产物、全部计划种子及复杂度缺口，补独立协议表；保留负结果、失败和未完成项。
- RunId：rgsp-acd-expansion-20261013-1053。

## EXP-SAMPLE-DIAGNOSIS-20261013 · 数据/指标审计与重训设计

- 用户要求检查SAMPLE低分原因，按现行表格基线训练方式分析设计，并两次要求继续。授权范围本轮完成数据与源码诊断、公开来源核对和独立设计；未自行启用新训练或调整队列。
- A/C独立任务目录以nice和单线程CPU检查1345张当前输入，0标签不一致；train/val/test均值125.957/126.082/158.257，边框均值119.971/120.153/157.499。18次已完成run重算Accuracy/Macro-F1与test一致，539样本和逐类支持数匹配；10关键源哈希与生产inventory一致。此为现有结果算术/源码核查，非神经网络重载推理验收。
- Full seed0 test2/539=0.3711%，source-val77.64%；GoogLeNet source-val96.27%、test2.23%，全部预测成BMP2/BTR70/M548。存在类别集中及源/目标分布差异，但尚无受控重训可确定各因素因果贡献。低分保留，不按高于随机作为指标真实性门槛。
- MLDA官方XML SHA fc0fae9cf57b6c77ae17c3b923b5bae6a0e8886c5fa18d619cc1db7e086cbbd8：uint16 amplitude/[0,1]、806/539、SGD.001、momentum.9、WD.0005、BS36、cosine10000更新；当前decibel8bit224、645/161/539、AdamW、最多1650更新不符。训练resize/初始化/精确归一化/选模/增强域batch组织未公开，不声称完全复现。旧constant LR设计保留但不能视作作者配置。裸R50=48.37%、ERM=91.78%、noise-only=74.68为不同来源行，不互相冒名。
- 官方SAMPLE commit1bae4156afcba7ad9c9f2eb743ed20675ed14fb1：README定义decibel=20log10(x+eps)、QPM=sqrt(x)，均缩放为uint8；C全2690+2690 PNG header核实L8。A下载两份官方MAT，complex_img均finite128x128complex128；可提取abs但不能据两文件声称全量已取/MLDA量化算子已核。官方verify.py有改写行为，仅静态读未执行。
- Inkawhich作者固定commit92a7bbc40e52bec6ede8f9be4a025915b3f9b5ad：QPM gray64中心crop，noise std.4/clamp，[-1,1]，带dropout.4的单通道R18，scratch/Adam.001/WD0/BS128/60e。不能混入MLDA协议或用普通R50冒充其方法；本轮未运行作者代码。
- DESIGN为PROPOSAL：用户BS16x4保留，原始幅度候选128/806source、SGDcosine10000，Full→Original→Matched075→R50，两明确增强组和预定3seed；另列表示因素组控制与物理BS36提案。逐图max量化、通道/选模等均明确本地选择，作者精确缩放未知。未来不得沿用旧convertRGB/255读uint16，也不得用实测全集统计拟合预处理。未部署、未按测试成绩择优。
- 11:15 A/B/C/D/F均可达；A/C/D/F已有训练，B瞬时空闲但既有ACD队列保留。仅A/C数据CPU任务，无本机模型、GPU训练/推理、安装、杀进程或持续监测。两份audit/两montage/两MAT与mat_probe共7文件本地SHA与服务器一致。新协议耗时尚未测，不沿用旧150epoch估计。
- 入口experiments/sample_diagnosis_20261013_v1/REPORT.md，SHA 728469de9996b3269eec4e94e8fb47b823c96f36e4bbff1819896a66278bf613；DESIGN.md SHA 2ed88a5f52b690804c334fbf727a6967ce50f5af209dd70a4a1df56d1f035a0f；完整证据audit_a、audit_c、code_audit、literature、official_mat_probe。RunId=sample-diagnosis-20261013-v1。
