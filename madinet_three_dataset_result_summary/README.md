# MaDiNet 三数据集方法对比结果汇总

生成日期：2026-06-23

## 1. 汇总目的

本文件夹用于保存 MaDiNet 论文三组数据集实验表的统一整理结果，并加入当前项目方法 **IG-DS-SSM / PFIM** 进行对比。

MaDiNet 论文包含三个主要数据集：

- SSDD
- SAR-ShipDataset
- SAR-AIRcraft1.0

本次整理将 MaDiNet 原文表格中出现的方法合并到一张统一对比表中，同时补充已有证据支持的模型参数量和 GFLOPs。

## 2. 文件说明

| 文件 | 说明 |
|---|---|
| `madinet_three_dataset_methods_comparison.tex` | 可编辑的 LaTeX 源文件 |
| `madinet_three_dataset_methods_comparison.pdf` | 已编译的 1 页横向 PDF 表格 |
| `madinet_three_dataset_methods_comparison_preview.png` | PDF 渲染预览图 |
| `madinet_three_dataset_methods_comparison_data.csv` | 表格核心数据的 CSV 版本 |
| `README.md` | 当前总结说明 |

## 3. 指标来源

检测精度指标主要来自 MaDiNet 论文源码：

- `downloads/madinet_paper/source/main_MaDiNet.tex`

对应表格为：

- SSDD：MaDiNet Table `tabel_ssdd_results`
- SAR-ShipDataset：MaDiNet Table `table_shipdata_results`
- SAR-AIRcraft1.0：MaDiNet Table `table_aircraft_results`

复杂度指标来自本项目已有本地记录：

- `database/open_source_ssdd_complexity_comparison_plan.csv`
- `reports/madinet_ds_ssm_complexity_report.md`

## 4. 当前关键结果

### MaDiNet 原文三数据集结果

MaDiNet 在原文中报告：

| Method | SSDD AP50 | SAR-ShipDataset AP50 | SAR-AIRcraft1.0 mAP50 |
|---|---:|---:|---:|
| MaDiNet | 99.0 | 97.5 | 90.5 |

这些值是 MaDiNet 原文的主 SOTA 参考。

### 我们当前方法结果

当前项目主线为 **IG-DS-SSM / PFIM**，也就是 PFIM 增强的 DS-SSM。

已验证 SSDD 结果：

| Method | Dataset | P | R | AP50 | AP50:95 | Params(M) | GFLOPs |
|---|---|---:|---:|---:|---:|---:|---:|
| IG-DS-SSM / PFIM | SSDD | 97.378 | 95.219 | 98.704 | 74.022 | 5.9648 | 13.2079 |

目前 **SAR-ShipDataset** 和 **SAR-AIRcraft1.0** 尚未完成我们方法的干净验证实验，因此在综合表中保留为空值。

## 5. 复杂度口径说明

表格中只填入已有证据支持的复杂度数值。

已本地测量的代表性方法：

| Method | Params(M) | GFLOPs | 说明 |
|---|---:|---:|---|
| YOLOv3 | 103.7541 | 283.3202 | 640 input, batch=1, fp32 |
| YOLOv5n | 2.6548 | 7.8401 | 640 input, batch=1, fp32 |
| YOLOv8n | 3.1572 | 8.8575 | 640 input, batch=1, fp32 |
| YOLOv5s | 9.1532 | 24.2129 | 640 input, batch=1, fp32 |
| IG-DS-SSM / PFIM | 5.9648 | 13.2079 | 640 input, batch=1, fp32 |
| MaDiNet official MambaSAR res4+res5 | 223.2648 | 345.2830 | 官方核心模块，不是完整 detector |

重要注意：

- MaDiNet 的复杂度数字是官方 `MambaSAR.py` 中 `res4+res5` 核心模块的本地测量结果。
- 该数值不能写成完整 MaDiNet detector 的总参数量或总 FLOPs。
- 论文中应写为 “MaDiNet official MambaSAR core module” 或在脚注中说明口径。

## 6. 推荐论文表述

安全表述：

- 我们方法在 SSDD 上达到 98.704 AP50 和 74.022 AP50:95。
- 我们方法参数量约 5.96M，GFLOPs 约 13.21。
- 与 MaDiNet 的 SSDD AP50 99.0 相比，我们方法 AP50 略低，但模型规模和计算量显著更轻。
- 当前工作更适合强调 “lightweight SAR-aware state-space detector” 和 “near-SOTA SSDD accuracy with lower complexity”。

不建议表述：

- 不要声称我们方法全面超过 MaDiNet。
- 不要声称完整 MaDiNet detector 的参数量/GFLOPs 已测量。
- 不要声称我们方法已经在 SAR-ShipDataset 或 SAR-AIRcraft1.0 上验证，除非后续补充实验。

## 7. 后续建议

优先补充实验：

1. 使用固定命令重新验证 PFIM ex27 的 SSDD 测试结果。
2. 在 SAR-ShipDataset 上测试 IG-DS-SSM / PFIM。
3. 如果论文范围从 ship detection 扩展到 SAR target detection，再考虑 SAR-AIRcraft1.0。
4. 若要写完整效率对比，继续测量 MaDiNet、DiffDet4SAR、DiffusionDet、YOLOX 等方法的完整 detector 复杂度。

