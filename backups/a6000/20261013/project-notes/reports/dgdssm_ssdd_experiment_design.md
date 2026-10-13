# SSDD 上基于 Gated DeltaNet-2 思想改进 DS-SSM 的实验设计

## 0. 当前限制说明

你提到的参考文件：

`C:/Users/13207/Downloads/GatedDeltaNet2_to_IG_DS_SSM_full_reference.md`

当前不在项目工作区 `D:\A-主机文件\Codex\Mamba-YOLO-ds-ssm` 内，因此我暂时没有读取其具体内容。本方案基于以下已有信息制定：

- 当前主线：`DS-SSM` 是核心创新，`PFIM` 是内部感知重要性引导机制；
- 当前最佳 SSDD 结果：`PFIM ex27`，mAP50-95 = `0.74022`；
- 当前代码结构：`gamma`、`write_gate`、`delta` 已经和 Gated DeltaNet-2 的 erase/write/delta 思路高度接近；
- 目标：在 SSDD 数据集上提升指标，尤其是 mAP50-95。

建议你后续把该 md 复制到：

`docs/01_candidate_designs/GatedDeltaNet2_to_IG_DS_SSM_full_reference.md`

然后再做一次逐条对齐审计。

---

## 1. 总体实验目标

目标不是推翻当前最优 `ex27`，而是在它的基础上做小步、可归因的结构改进。

当前最佳：

| 实验 | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| `PFIM ex27` | 0.97378 | 0.95219 | 0.98704 | **0.74022** |

本轮实验目标：

1. 保持 DS-SSM 主体结构不变；
2. 保持 batch size = 32；
3. 不引入过大的新模块；
4. 将 Gated DeltaNet-2 的 erase/write 解耦思想融入当前 `gamma/write_gate/delta`；
5. 争取 mAP50-95 从 `0.74022` 提升到 `0.742+` 或更高；
6. 即使未提升，也要得到清楚的消融结论。

---

## 2. 当前结构与 Gated DeltaNet-2 思想的对应关系

当前 DS-SSM/PFIM 里已有：

| 当前结构 | 当前作用 | Gated DeltaNet-2 对应思想 |
|---|---|---|
| `gamma` | 控制旧状态保留/衰减 | erase / retain gate |
| `write_gate` | 控制当前输入写入强度 | write gate |
| `delta` | 控制状态转移步长 | transition / update rate |
| `PFIM importance` | 给 gamma/write/delta 提供目标重要性 | content-aware gating signal |

所以最自然的改进方向是：

**不要另起炉灶，而是把当前 PFIM-guided gamma/write/delta 改成更清晰的 decoupled erase/write/delta 结构。**

---

## 3. 建议新模块命名

建议候选模块命名：

`DG-DS-SSM`

全称：

`Decoupled-Gated De-Speckle Selective State Space Model`

论文里可以表达为：

> 在 DS-SSM 的基础上，引入 erase/write 解耦门控机制，使状态保留和新信息写入分别由独立分支控制，从而缓解 SAR 图像中噪声抑制与目标保持之间的冲突。

---

## 4. 代码修改设计

### 4.1 不建议直接修改原始 `mamba_yolo.py`

为了保护当前最优 `ex27`，建议新增文件：

`ultralytics/nn/modules/mamba_yolo_dg_dsssm.py`

或者：

`ultralytics/nn/modules/mamba_yolo_ex30_dg_dsssm.py`

推荐使用第二个，便于实验追踪：

`mamba_yolo_ex30_dg_dsssm.py`

从当前 `ex27` 对应实现复制，然后只改 DS-SSM/PFIM 部分。

### 4.2 新增配置文件

新增 5 个 YAML：

```text
ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DG-DSSSM-SSDD-ex30-baseline.yaml
ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DG-DSSSM-SSDD-ex31-erase.yaml
ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DG-DSSSM-SSDD-ex32-write.yaml
ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DG-DSSSM-SSDD-ex33-erase-write.yaml
ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DG-DSSSM-SSDD-ex34-erase-write-delta.yaml
```

### 4.3 新增模块注册

需要检查并修改：

```text
ultralytics/nn/modules/__init__.py
ultralytics/nn/tasks.py
```

目标是让 YAML 能识别新的模块类，例如：

```python
DG_DS_VSSBlock
```

### 4.4 核心代码改法

当前 `ex27` 大致是：

```python
importance = self.pfim(x_tilde, x_bar, hw_shape=hw_shape)
imp_center = importance - 0.5

gamma_logits = -alpha * sigma_n * energy + beta_i * imp_center
gamma = min_retain + (1 - min_retain) * sigmoid(gamma_logits)

write_gate = 1 + tanh(rho_i) * imp_center
u_write = u * write_gate

delta_logits = delta_proj(u) + eta_i * importance
```

建议改成更明确的解耦：

```python
importance = self.pfim(x_tilde, x_bar, hw_shape=hw_shape)
imp_center = importance - 0.5

erase_logits = -alpha * sigma_n * energy + beta_e * imp_center
retain_gate = min_retain + (1 - min_retain) * sigmoid(erase_logits)

write_logits = rho_w * imp_center + content_proj(u)
write_gate = 1 + write_scale * tanh(write_logits)
write_gate = clamp(write_gate, min=0.1, max=2.0)

delta_logits = delta_proj(u)
if use_delta_gate:
    delta_logits = delta_logits + eta_d * importance

u_write = u * write_gate
u_damped = u_write * retain_gate
```

### 4.5 关键原则

为了避免小数据集过拟合，本轮不要让 PFIM 输出太多复杂 map。

优先采用：

- 一个 `importance`；
- 独立可学习参数 `beta_e/rho_w/eta_d`；
- 可选轻量 `content_proj(u)`，先不加复杂 MLP。

如果第一轮有效，再尝试 multi-head PFIM：

```python
importance_erase, importance_write, importance_delta
```

---

## 5. 实验矩阵

本轮建议至少做 5 个实验，全部 batch size = 32，epochs = 300，imgsz = 640。

| 实验编号 | 名称 | 改动 | 目的 | 预期 |
|---|---|---|---|---|
| ex30 | `baseline-ex27-rerun` | 原 ex27 结构重新跑一遍 | 校准当前服务器/随机性 | 应接近 0.740 |
| ex31 | `erase-only` | 只改 `gamma` 为 decoupled erase/retain gate | 看 erase 机制是否提升定位质量 | 可能提升 mAP50-95 |
| ex32 | `write-only` | 只改 `write_gate`，保留原 gamma/delta | 看写入门控是否提升召回 | 可能提升 R |
| ex33 | `erase-write` | 同时解耦 erase + write | 核心候选结构 | 最值得关注 |
| ex34 | `erase-write-delta` | erase/write 解耦 + delta gate | 完整 DG-DS-SSM | 可能最好，也可能不稳定 |

可选第 6 个：

| 实验编号 | 名称 | 改动 | 目的 |
|---|---|---|---|
| ex35 | `multi-importance` | PFIM 输出 3 个 importance | 验证分支专用 importance 是否优于共享 importance |

建议第一轮先不做 ex35，除非服务器资源充足且你希望更激进。

---

## 6. 服务器分配建议

你给了 5 台服务器：

| 服务器 | IP | 建议任务 |
|---|---|---|
| 3090-5 | `172.26.150.72` | ex30 baseline rerun |
| 3090-4 | `172.26.133.16` | ex31 erase-only |
| 3090-3 | `172.26.151.143` | ex33 erase-write |
| 3090-1 | `172.26.150.36` | ex34 erase-write-delta |
| 1080Ti-2 | `172.26.193.188` | ex32 write-only 或 smoke test |

注意：

- 1080Ti 显存比 3090 小，`batch_size=32` 可能有 OOM 风险。
- 如果 1080Ti 跑不动 T 模型 bs32，可以改为只做 smoke test，或用 2 卡 DDP 每卡 16。
- 关键实验建议优先放在 3090 上：ex31、ex33、ex34。

---

## 7. 推荐训练命名

统一命名格式：

```text
ssdd_DG-DSSSM_T_300e_32bs_ex30_baseline_20260608
ssdd_DG-DSSSM_T_300e_32bs_ex31_erase_20260608
ssdd_DG-DSSSM_T_300e_32bs_ex32_write_20260608
ssdd_DG-DSSSM_T_300e_32bs_ex33_erase-write_20260608
ssdd_DG-DSSSM_T_300e_32bs_ex34_erase-write-delta_20260608
```

不要再用冒号 `:`，避免后续 Windows 归档麻烦。

---

## 8. 训练命令模板

单卡 3090 推荐：

```bash
cd /home/zhangji/Mamba-YOLO
conda activate mamba-yolo

CUDA_VISIBLE_DEVICES=0 python mbyolo_train.py --task train \
  --data datasets/SSDD_YOLO/ssdd.yaml \
  --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DG-DSSSM-SSDD-ex31-erase.yaml \
  --batch_size 32 \
  --imgsz 640 \
  --epochs 300 \
  --device 0 \
  --workers 4 \
  --amp \
  --project output_dir/ssdd \
  --name ssdd_DG-DSSSM_T_300e_32bs_ex31_erase_20260608
```

双卡或 1080Ti DDP 备选：

```bash
cd /home/zhangji/Mamba-YOLO
conda activate mamba-yolo

python mbyolo_train_ddp_ex27.py \
  --data datasets/SSDD_YOLO/ssdd.yaml \
  --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DG-DSSSM-SSDD-ex32-write.yaml \
  --batch_size 32 \
  --imgsz 640 \
  --epochs 300 \
  --device 0,1 \
  --workers 4 \
  --project output_dir/ssdd \
  --name ssdd_DG-DSSSM_T_300e_32bs_ex32_write_20260608
```

---

## 9. 指标记录规则

每个实验都必须记录：

| 字段 | 内容 |
|---|---|
| experiment_id | ex30/ex31/ex32/ex33/ex34 |
| server | IP 和服务器编号 |
| code_version | 本次代码快照 |
| config | YAML 路径 |
| batch_size | 32 |
| epochs | 300 |
| best_epoch | 从 results.csv 解析 |
| P | best row |
| R | best row |
| mAP50 | best row |
| mAP50-95 | best row |
| best.pt | 路径 |
| last.pt | 路径 |
| notes | 是否 NaN/OOM/中断 |

本地更新目标：

```text
database/dg_dsssm_ssdd_experiment_plan.csv
database/dg_dsssm_ssdd_results.csv
reports/dg_dsssm_ssdd_experiment_results.md
```

---

## 10. 成功标准

### 强成功

任一实验：

```text
mAP50-95 > 0.74022
```

说明 Gated DeltaNet-2 式解耦确实优于当前 ex27。

### 中等成功

任一实验：

```text
mAP50-95 >= 0.738
```

并且 P/R 有明显改善，例如：

- Recall 提高；
- mAP50 接近或超过 0.987；
- 曲线更稳定。

说明该方向值得继续微调。

### 失败但有价值

如果 ex31/ex32/ex33/ex34 都低于 0.735，但训练稳定：

- 说明当前 ex27 已经较优；
- Gated DeltaNet-2 式改法可能不适合 SSDD 小数据；
- 可以把该方向作为负结果或未来工作，不影响当前论文主线。

---

## 11. 风险与应对

| 风险 | 表现 | 应对 |
|---|---|---|
| gate 过强 | loss NaN，mAP 很低 | clamp gate，降低初始化 |
| 过拟合 | train loss 降，val mAP 不升 | 减少参数，不做 multi-head |
| Recall 升但 mAP50-95 降 | 定位质量下降 | 降低 write scale 或 delta gate |
| DDP 不稳定 | unused parameter 报错 | `find_unused_parameters=True` |
| 1080Ti OOM | 显存不足 | 改 DDP 每卡 16 或只做 smoke |

---

## 12. 推荐执行顺序

第一轮并行：

1. ex30 baseline rerun；
2. ex31 erase-only；
3. ex32 write-only；
4. ex33 erase-write；
5. ex34 erase-write-delta。

第二轮根据结果：

- 如果 ex31 最好：继续调 `beta_e/min_retain`；
- 如果 ex32 最好：继续调 `rho_w/write_scale`；
- 如果 ex33 最好：固定为 DG-DS-SSM 主候选；
- 如果 ex34 最好：说明 delta gate 有价值，补 delta 消融；
- 如果都不如 ex27：回到 ex27 主线，不继续复杂化。

---

## 13. 我建议的优先级

最值得优先跑：

1. **ex33 erase-write**
2. **ex31 erase-only**
3. **ex34 erase-write-delta**
4. ex32 write-only
5. ex30 baseline rerun

原因：

- ex33 最贴近 Gated DeltaNet-2 的核心思想；
- ex31 可以验证“旧状态擦除/保留”是否是主要贡献；
- ex34 是完整增强，但也最容易不稳定；
- ex30 是必要校准，但如果资源紧张可以只用历史 ex27 作参考。

---

## 14. 下一步需要你确认

正式改代码和上服务器前，需要你确认：

1. 是否允许基于 `ex27` 新增 `mamba_yolo_ex30_dg_dsssm.py`；
2. 是否确认本轮全部 batch size = 32；
3. 1080Ti 是否也必须跑完整 300 epoch，还是只做 smoke/备用；
4. 是否五台服务器可以同时修改项目代码并启动训练；
5. 是否优先跑 ex31-ex34，还是必须包含 ex30 baseline rerun；
6. 是否把这个方向命名为 `DG-DS-SSM`。

确认后再进入下一阶段：代码实现、分发服务器、并行训练和结果回收。
