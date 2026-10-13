# RGSP 实验补充与服务器执行记录

日期：2026-10-08，Asia/Shanghai。

本记录区分代码验证、正式训练运行中、最终结果完成。正文实验补充见：

[RGSP_EXPERIMENTS_SUPPLEMENT_20261008.md](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/paper_draft/RGSP_EXPERIMENTS_SUPPLEMENT_20261008.md)。

## 1. 实际启动

正式 campaign：rgsp_controls_20261008_v3。

2026-10-08 12:30:15 左右，三个独立单 GPU 队列已启动。每个队列包含预先固定的 12 个配置，seed 分别为 0、1、2，共 36 个正式 run，最多各 150 epoch，patience=30。当前首先训练 Original，不代表其他配置已完成。

| 主机 SSH 别名 | 实际 GPU | seed | 队列 PID | 首个训练子进程 PID |
|---|---|---:|---:|---:|
| powerleader-a6000-f | GPU0，RTX A6000 | 0 | 2274058 | 2274063 |
| powerleader-a6000-f | GPU1，RTX A6000 | 1 | 2274059 | 2274062 |
| powerleader-a6000-c | GPU0，RTX A6000 | 2 | 3544342 | 3544345 |

PID 是启动时证据，不是永久进程标识。后续检查必须同时核对命令行、任务目录和 status.json，不能按旧 PID 盲目操作。

初次运行确认：三卡 GPU 利用率约 98%–100%，各训练子进程约 15,698 MiB 显存。此处是启动时点观察，后续实时状态优先。

12:32:44 的再次检查：三个 Original 均已完成 epoch5，三个 seed 的 box/cls/dfl 训练损失均为有限值，三卡 GPU 利用率均为 100%。已回收此时的 protocol、数据清单、runtime_audit、results.csv 和逐轮审计。这个证据确认训练在推进，**不是最终性能或整个 36-run 矩阵已经完成**。

远端根目录（F、C 相同）：

~~~text
/home/powerleader/ig-ds-ssm/tasks/rgsp_controls_20261008_v3/
├── code/                         独立源码副本，不覆盖原 work_train_single
├── code_frozen.tar.gz            启动源码包
├── seed0/ 或 seed1/ 或 seed2/
│   ├── manifest/                数据、划分、代码及配置哈希
│   ├── data/                    本轮独立数据副本与缓存
│   ├── runs/<variant>/          权重、args、训练结果及控制审计
│   ├── logs/                    train/eval 日志
│   ├── evaluations/             最终 test 评估工作文件
│   ├── status.json              队列状态与已完成配置
│   └── results_partial.json     已完成测试结果，不等于全矩阵结果
└── seed0.queue.log 等             队列日志
~~~

队列行为：单配置训练成功后，评估验证集选出的 best.pt 在固定官方 test 上的指标，再进入下一项。遇到异常则该队列失败退出，不抢占其他 GPU、不自动更改超参数、不静默丢弃失败 seed。进程以脱离 SSH 的有限队列启动，不依赖当前聊天持续打开。

未创建系统计划任务、开机自启或定时监控。尚未产生的成绩不填入论文。

## 2. 资源盘点与选择

初次盘点为当日约 11:58，覆盖项目关联的 A/B/C/D/F 五台服务器；未扫描未登记网段。

| 目标 | 初检资源 | 本轮处理 |
|---|---|---|
| A | 1×A6000，无计算进程；约 281 GB 可用磁盘 | 不启动训练；未找到可直接验证的第二数据集协议 |
| B | 1×A6000，无计算进程；约 655 GB 可用磁盘 | 备用；已查相关数据目录 |
| C | 1×A6000，ComfyUI 占少量显存；约 885 GB 可用磁盘 | 用户明确授权后终止 ComfyUI，投入 seed2 |
| D | 1×A6000，初检空闲；约 176 GB 可用磁盘 | 后续连接超时，本轮不作为训练目标 |
| F | 2×A6000，无计算进程；约 278 GB 可用磁盘 | GPU0/GPU1 分别运行 seed0/seed1 |

C 上被结束的进程为 PID 3539324，进程目录核对为 /home/powerleader/project/ComfyUI，命令为 ComfyUI 环境中的 main.py、端口 8188。仅发送 SIGTERM，结束后核验该进程消失、GPU 无计算进程。没有删除其文件，没有修改自启动或服务配置。

所用环境：已有 Python 3.10.16、PyTorch 1.13.1+cu117、CUDA 11.7。未安装软件，未在本机运行模型训练/推理/测试，也未启动本机 4050 服务。

GPU UUID：

- F0：GPU-9d27eec3-fd30-8534-8341-b983bea3832b
- F1：GPU-ce57e76d-ee2e-ff91-dbdd-ec0938874c40
- C0：GPU-e5f8a5d2-306d-582e-c926-f795df943b1b

## 3. 训练前查到的关键问题

### 3.1 旧 PFIM-only 不是严格关闭 R/E 的控制

直接加载旧三个 best.pt，哈希与旧结果表一致，lambda/alpha 分别为：

- seed0：0.0219879150 / -0.0108566284；
- seed1：0.0011110306 / -0.0126419067；
- seed2：-0.0000919104 / -0.0191802979。

构造函数设为零且冻结，但训练器重新解冻。故旧 PFIM-only 的“独立贡献”解释撤回；保留其真实历史成绩，不宣称整个历史实验表全部无效。

证据：[F 端控制与历史权重审计](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/experiments/rgsp_controls_20261008_v1/evidence/f/controls.json)。

### 3.2 内容重复与测试重叠

现存 SSDD 副本共 1160 张图、1135 个不同字节内容、22 组重复；10 组跨官方 train/test，涉及 12 张训练副本和 10 张 test 图。同内容组的标注文件均存在差异。

证据：[重复图清单与标注差异](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/experiments/rgsp_controls_20261008_v1/evidence/f/data_duplicates.json)。

正式采用内容防泄漏轨道：保留官方 test232；从本轮开发集排除上述 12 张训练副本；余下 916 张、903 个内容组，按组划分为 train730/val186。同内容不得跨 train/val/test；所有原始数据文件保持不变。

三个 seed 的来源清单 SHA-256 相同：

~~~text
6841926f151e51d38e2ff9c837a1f5524acd9da4d64c13d7dc0a9c9ffd3f10f3
~~~

这不是近重复/场景独立性证明；新结果不能与旧 742/186/232 轨道混算三 seed 均值。

### 3.3 模型构建探针与评估进程退出

- 模型推算 stride 的空白前向会改变 BN 运行统计；本轮对所有 scratch 配置统一重置，再验证共同初始张量。
- 旧实际 cos_lr=false；正文按线性衰减描述。
- 独立评估在 workers=4 时出现 pin-memory daemon 的解释器退出异常，虽然已写出 JSON，仍判为进程失败而非正式验收成功。workers=0 的独立验证重试正常退出；最终队列固定测试 workers=0，指标设置未改。未安装或升级 PyTorch。

## 4. 修改与验证

增量修改模块：

[mamba_yolo.py](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/source_code/work_train_single/ultralytics/nn/modules/mamba_yolo.py)。

新增入口：

- [run_rgsp_controls.py](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/source_code/work_train_single/tools/run_rgsp_controls.py)：数据清单、分组划分、源码冻结、有限队列和结果汇总。
- [train_rgsp_control.py](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/source_code/work_train_single/tools/train_rgsp_control.py)：明确超参数，训练时审计不可训练常量和有限损失。
- [validate_rgsp_controls.py](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/source_code/work_train_single/tools/validate_rgsp_controls.py)：数值、梯度、开关语义、初始化和旧权重审计。
- [audit_ssdd_content.py](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/source_code/work_train_single/tools/audit_ssdd_content.py)：只读内容重复检查。
- [launch_rgsp_lane.sh](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/source_code/work_train_single/tools/launch_rgsp_lane.sh)：单 GPU 有限队列入口。

旧 Sweep 不传 control 时保持原行为；关闭参数在新控制中保存为 buffer。未修改旧训练器的通用解冻规则，未批量重写旧模块与 checkpoint。

已完成验证：

1. F/C 的 fused selective scan 与参考实现输出、梯度比较 PASS。
2. 11 个匹配控制的倍率/步长语义、两步真实优化、常量保持与有限梯度检查 PASS。
3. 12 个完整模型的 YAML 解析、P3 实际块数量与共同初始张量检查 PASS。
4. F 上全部 12 配置完成 32/16 数据、2 epoch 的端到端短训练；C 上 Original/Guidance-only/Full 也完成。
5. 新防泄漏划分下 F Full、C Guidance-only 的两轮短训练及重载 PASS。
6. 独立评估 CLI 在验证子集、workers=0 下完成并正常退出；没有将短训练指标用作科学结果。
7. 三 seed 数据内容清单一致、730/186/232 数量符合、跨集合字节级相同图像交集为空。

预检中保留的失败记录：初始化测试脚本引用了可变 state_dict，已改为 clone；探针 BN 统计差异按统一重置处理；global 是 Python 保留字，配置参数改为显式字符串字面量；数据重叠由分组协议解决；评估线程退出改为 workers=0。失败日志和早期目录保留，不冒充正式运行。

最终源码包：

[code_frozen.tar.gz](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/experiments/rgsp_controls_20261008_v3/code_frozen.tar.gz)

SHA-256：

~~~text
9e814008b4beb072aabf3cd1e71afd617e612f6fd8cebbf560e686391099985b
~~~

两台服务器收到的包哈希与本地一致。正式源代码不可在运行期间就地修改；需要改变实验时新建版本。

## 5. 尚未完成

- 36 个正式 run 的训练和 test 指标：运行中，当前不能填最终主表。
- 跨 seed 最终均值、样本标准差、配对差值与两组四格交互：需等完整结果。
- 第二数据集：没有取得可核验的完整来源/划分，本轮未启动。
- 公平外部轻量基线、batch=1 延迟、完整自定义算子复杂度、机制诊断和困难案例：未完成。
- 原始整篇双栏 LaTeX 稿没有被此文件覆盖；本轮交付为可合并的 Markdown 实验章节。

下一次接手：先查询这三个队列的 status、实际进程和 epochs，收集现有结果；不要重复启动相同任务，不要把 v1/v2 短训练目录当正式成绩。
