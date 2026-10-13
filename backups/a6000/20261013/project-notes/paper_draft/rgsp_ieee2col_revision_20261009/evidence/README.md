# 证据范围

frozen_ssdd/与frozen_sadd/的ultralytics/nn/modules/mamba_yolo.py分别原样提取自本项目
experiments/rgsp_controls_20261008_v3/code_frozen.tar.gz及
experiments/rgsp_sadd_controls_20261008_v1/code_frozen.tar.gz。
两份模块文件SHA256相同，见REVISION_NOTES；这是定义核对片段，不是完整训练环境。
frozen_ssdd/full.yaml原样复制自该轨道f/seed0/manifest/configs/full.yaml。

recovery/sadd_seed{0,1,2}_verification.json原样复制自
experiments/rgsp_recovery_20261008_v1/evidence/sadd_seed{0,1,2}/verification.json，
是恢复前核验摘要。不要把其中gpu_training_executed=false当作后续未完成，也不要把PASS当作最终训练完成。
后续完成状态与权重核验来自2026-10-09的reports/SADD_RESULTS_SSDD_RESTART_20261009.md。
回收的逐项test比例值和权重SHA见../supplementary/sadd_seed*.json。

本轮只复核本地已回收证据及文档，未连接服务器查询实时状态、未复制大权重、未运行模型。
