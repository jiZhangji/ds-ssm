# BASELINE_PROTOCOL.md — 公平基线协议

> 项目：Mamba-YOLO / IG-DS-SSM / PFIM　日期：2026-10-01
> 目的：把"公平比较"从口号变成可核查的冻结规则，杜绝历史账本里的协议不齐问题。

---

## 1. 基线定义

### 1.1 主基线：plain Mamba-YOLO-T（`baseline` 变体）

- 结构：VMamba-Tiny 编码器（`VMambaTinyBackbone`）+ 全部 `VSSBlock`，**不含任何 DS-SSM / PFIM 组件**。
- 生成方式：`run_sarmix_proxy.model_payload("baseline", 0, "full")` → `selected=∅`，所有 P2/P3/P4/P5 层均为 `VSSBlock`，state_dim=0。
- 这是"加 DS-SSM/PFIM 之前的对照"，是 Q1/Q2 的因果锚点。

### 1.2 消融对照（均为 P3 接入；baseline 与 DS 变体参数**近似**但非严格对齐，见 §4.1）

| 变体 key | 接入类 | DS-SSM 残差修正 | 能量门控 | PFIM | 回答的问题 |
|---|---|---|---|---|---|
| `baseline` | VSSBlock | ✗ | ✗ | ✗ | 锚点 |
| `ds_only_p3` | DS_VSSBlockSweepNoPFIM | ✓ | ✓ | ✗ | Q1（DS-SSM 增益） |
| `pfim_only_p3` | DS_VSSBlockSweepPFIMOnly | ✗(lambda_s=0) | ✗(alpha=0) | ✓ | Q2（PFIM 单独增益） |
| `full_p3` | DS_VSSBlockSweep | ✓ | ✓ | ✓ | 完整方法 |
| `global_p3` | DS_VSSBlockSweepPFIMGlobal | ✓ | ✓ | ✓(image-wise) | Q3（token vs image） |

> 关键语义（据 mamba_yolo.py 源码，禁止误述）：
> - `pfim_only` 是 `lambda_s=0, alpha=0` 且**冻结 requires_grad**，即关掉残差修正与能量门控，但保留 PFIM 三路调节（gamma/write/delta）。
> - `global` 是 `pfim_tokenwise=False`，PFIM 用图像级（非 token 级）重要性，参数与 full 匹配。

---

## 2. 公平性规则（硬约束，逐条可核查）

### 2.1 同一初始化

- **scratch**：`--weights none`（所有变体一致），不做任何预训练 warm-start。
- 禁止：给 ours 用预训练、给 baseline 用 scratch（历史 SAR-AIRcraft 的不公平点，本阶段杜绝）。

### 2.2 同一数据划分

- SSDD 用**同一个 frozen split**：official train 928 张按固定 seed 切 186 张为 val、742 张为 train；official test 232 张**全程封存**，仅最终评估一次。
- 所有变体共享同一 `SSDD.yaml`（sha256 冻结）+ 同一 `ssdd_split/snapshot.json`。
- **禁用** `--ssdd-full-train`（会把 test 当 val，test-as-val 泄漏）。

### 2.3 同一训练步数 / 输入 / 优化器 / 调度

| 项 | 固定值 | 说明 |
|---|---|---|
| imgsz | 640 | 所有变体 |
| epochs | 150（patience 30 early stop） | 统一 |
| batch | 16 | 主 batch（历史 scratch 正向协议）；BS32 作单独对照 |
| optimizer | AdamW | 统一 |
| lr0 | 2e-4 | 统一 |
| lrf | 0.01 | 统一 |
| weight_decay | 5e-4 | 统一 |
| nbs | 64 | 统一 |
| warmup_epochs | 3.0 | 统一 |
| warmup_bias_lr | 0.1 | 统一 |
| module_warmup_epochs | 0 | 统一（不特殊照顾新增模块） |
| scheduler | cos_lr（runner 默认） | 统一 |
| augmentation | mosaic 0.3 / close_mosaic 15 / degrees 180 / translate 0.1 / scale 0.3 / flipud 0.5 / fliplr 0.5 / hsv_h 0 / hsv_s 0 / hsv_v 0.1 | 统一 |
| precision | 默认（fp32，torch 1.13.1 老环境不冒进 amp） | 统一 |
| deterministic | `--deterministic` | 统一 |

### 2.4 同一选模规则

- **早停 + 最优 checkpoint 选择**：一律用 `weights/best.pt`（runner 在 val 上按 mAP50:95 选 best），评估只对 best.pt 跑 official test。
- **禁止**基于 official test 的任何选模、调参、早停。
- baseline 与 ours 用**完全相同**的 best 选择逻辑。

### 2.5 同一评估协议

- 评估只在 official test（232 张）上跑一次，`evaluate_sarmix.py --split test`。
- 指标：AP50 / AP75 / AP50:95（`--subset SSDD`）。
- 评估 batch 统一 16、imgsz 640、device 统一。

---

## 3. 多 seed 协议

- seed ∈ {0, 1, 2}，每个 seed 重新 shuffle 生成独立 split（seed 固定进 `snapshot.json`）。
- 每 seed 下，5 个变体共享**同一 seed 的 split**（pairing 在 seed 内成立）。
- 报告：individual（每 seed 每变体 AP）+ mean±std + paired Δ(ours − baseline)。
- 统计：3 seed 配对的 mean Δ + 逐 seed 方向一致性（≥2/3 同号即"稳定正向/负向"）。

---

## 4. 复杂度账本（公平性的另一面）

- 对 `full_p3` 与 `baseline` 各测 Params / GFLOPs / FPS（同输入 640、同 device、同 batch）。
- 任何 AP 增益必须与参数量/FLOPs 增量并列呈现，禁止"只报精度不报成本"。
- 测量工具：优先用项目已有 `tools/benchmark_*`；无则现场用 `thop`/`torch.profiler` 记录方法并写进 RESULTS。

### 4.1 实测账本（2026-10-01，best.pt + imgsz=640，seed0）

| 变体 | Params | GFLOPs | FPS | inference(ms) |
|---|---:|---:|---:|---:|
| baseline（VSSBlock） | 5,985,547 | 13.61 | 151.1 | 6.62 |
| full_p3（DS_VSSBlock+PFIM） | 5,920,441 | 13.15 | 164.2 | 6.09 |
| ds_only_p3（DS_VSSBlock，无 PFIM） | 5,918,104 | 13.12 | 166.7 | 6.00 |
| pfim_only_p3（仅 PFIM） | 5,920,441 | 13.15 | 163.8 | 6.11 |
| global_p3（PFIM image-wise） | 5,920,441 | 13.15 | 163.2 | 6.13 |

**参数不对齐的事实（须在论文如实披露，不得掩盖）**：
- `baseline` 与 DS 变体 **参数量并不完全相同**（5.99M vs 5.92M，差 6.7 万 ≈ 1.1%）。
- 根因（mamba_yolo.py 源码）：VSSBlock 用 `mlp_ratio=4.0` + SS2D `ssm_ratio=2.0` 膨胀 + LSBlock + RGBlock；DS_VSSBlock 用 `mlp_ratio=2.0` + DeSpeckleSSM（state_dim=16，无 rank 膨胀）。
- 这是**方法本身的轻量化设计**（DS-SSM 替换 backbone 的 VSSBlock 时连带缩小 MLP），不是人为调参制造优势，但也不是严格"参数对齐"的公平消融。
- full/pfim_only/global 三者 Params 完全一致（5,920,441）：PFIM 仅是 gamma/write/delta 三路标量调节，参数可忽略；ds_only 少 2,337 参数（即 PFIM 的标量参数量）。
- 结论措辞须诚实：DS 变体 **更轻更快**（−1.1% Params、−3.4% GFLOPs、+8.6% FPS），AP 增益须与这一成本优势并列报告，并明确"轻量化来自架构替换（含 MLP 缩小），非纯 SSM 机制增益"。

---

## 5. 禁止事项（红线清单）

1. 禁止 test-as-val（`--ssdd-full-train`）。
2. 禁止基于 official test 调参/选模/早停。
3. 禁止 ours 与 baseline 用不同 batch/optimizer/lr/imgsz/scheduler/augmentation。
4. 禁止 ours 用预训练、baseline 用 scratch（或反之）。
5. 禁止只报最优 seed、删失败/负结果 run。
6. 禁止换评价协议（换 IoU 阈值、换 NMS、换输入分辨率只在最后报告，不作"提升"证据）。
7. 禁止把"历史复现"（固定 TEST 的复评、warm-start、不同结构混用）与"投稿因果实验"混入同一张主表。

---

## 6. 可复现性冻结清单（每 run 必须落盘）

| 产物 | 内容 | 落盘位置 |
|---|---|---|
| `protocol.json` | dataset sha256 + config sha256 + code sha256 + 全部超参 + seed | `<project>/<exp>-manifest/protocol.json` |
| `ssdd_split/snapshot.json` | official_train/val/test 图数 + seed | `<project>/<exp>-manifest/datasets/ssdd_split/snapshot.json` |
| `train.txt` / `val.txt` | frozen 划分文件列表 | 同上目录 |
| `configs/*.yaml` | 每变体 model config（含 sha256） | `<project>/<exp>-manifest/configs/` |
| `best.pt` / `last.pt` | 权重 | `<project>/<exp>_<dataset>_<variant>/weights/` |
| `eval_test.json` | official test 评估结果 | `<project>/<exp>_<dataset>_<variant>/eval_test.json` |
| `results.csv` | 训练曲线 | `<project>/<exp>_<dataset>_<variant>/results.csv` |

---

## 7. 与历史账本的关系（诚实划界）

- 历史结果（PROJECT_AUDIT.md §4）均为 Partially Confirmed 或 Mentioned Only，**不进投稿主表**，仅作"探索轨迹"引用。
- 本协议产出的 3-seed 配对结果才是"投稿主表"唯一来源。
- 若主表结果与历史方向矛盾（如 SSDD BS16 正向 vs BS32 反向），如实并列，标注 batch 敏感性，不掩盖。

---

## 8. 待定项（需在实验启动前拍板，低风险默认值已给）

| 待定项 | 默认值 | 备注 |
|---|---|---|
| 主 batch | 16 | 历史 scratch 正向协议；BS32 作对照 |
| val_count | 186 | 928−186=742，吻合交接文档 742/186/232 |
| 是否跑 global_p3 进主表 | 是（作为 Q3 佐证） | 参数匹配，成本低 |
| 是否跑 BS32 对照 | 若 15 run 顺利则追加 | 单独标注，不混入主表 |
