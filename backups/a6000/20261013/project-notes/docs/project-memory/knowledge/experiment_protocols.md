# Experiment Protocols and Interpretation

## Evidence hierarchy

1. **Matched controlled**: same code snapshot, split, initialization, optimizer, batch, augmentation, epoch count and evaluator.
2. **Historical archive**: real logs/checkpoints but differing code or protocol.
3. **Author-reported external**: paper or public table value not reproduced locally.
4. **Plan only**: scripts/configs exist, but no result file or verified run.

Only level 1 should support causal claims in the main table. Levels 2-3 are contextual and must be labeled. Level 4 must not be presented as a result.

## Detection

SSDD development uses 742 train / 186 val / 232 official test in the H100 record. Final proposed protocol merges train+val to 928 train and evaluates once on 232 official test after structure selection. Main metrics are P, R, AP50, AP75, AP50:95; report Params, GFLOPs and FPS/latency under one profiler.

SSDD has two不可混用的历史口径：历史 928-train/300-epoch/bs32/lr1e-3 验证集选模（ex27 `0.74022` 属此类）与当前 742/186/232、150-epoch scratch 受控协议；历史权重另有固定官方 TEST 复核值（ex27 `0.714454`）。引用任何 SSDD 数字必须标注 split、epoch、batch 与初始化。

SRSDD-v1.0 原生为 OBB（666 图、2884 对象、6 类）；HBB 转换结果不得与 OBB 发表值直接比较；paper-aligned HBB 轨道用全 532 train/134 test、fixed-last-epoch。AIR-SARShip-2.0 的 paper-aligned 轨道为 Bao et al. 2021 的 210 train+val/90 test、512 裁剪 256 重叠。

OGSOD controlled evidence uses 1024 input, 300 epochs, global BS64, two H200, AdamW and FP32. Validation selects `best.pt`; TEST is held out. Do not select the best seed.

## Segmentation

FUSAR paper-aligned protocol uses four classes and WuHan_03_02 official test scene. Compare mIoU, FWIoU, mPA, Dice and OA; per-class IoU is necessary because road collapse can be hidden by large vegetation/water classes. OpenEarthMap-SAR records effective BS64 as physical BS32 + accumulation2, not physical BS64.

## Classification

SAR-ACD should report Accuracy and Macro-F1, not mAP. Its corrected classification head uses the P3 feature path; old unconnected-head warmup results must remain separate. SAMPLE is a closed-set accuracy comparison and is not directly comparable to class-incremental RSPC AIA results.

## Change detection

Ottawa, Yellow River and Farmland report Precision, Recall, F1, IoU and OA. BS16/32/64/96 runs are independent; batch-size changes are not interchangeable evidence. Farmland's positive result is mainly recall/F1 recovery from a weak baseline; Yellow River is a negative result and must remain visible.

## External baselines

Reproduce only a small set of key methods with usable code/config/weights. Keep a separate author-reported table for methods whose split, task format or implementation is unavailable. A local result that differs from a paper value is not automatically a bug; first audit protocol, labels, pretraining, evaluator and metric definition.
