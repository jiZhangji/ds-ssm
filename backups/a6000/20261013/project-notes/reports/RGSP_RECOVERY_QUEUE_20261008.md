# SSDD / SADD 消融中断恢复与后置队列

事件与核验：2026-10-08 19:56–20:08，Asia/Shanghai。用户当前授权：“你把这些任务复活一下，然后排队放在后面”。

## 已实施的状态

五条独立恢复队列已启动为 `waiting_predecessor`，不是 GPU 训练已经恢复。原 C 上的 SSDD seed2 仍继续运行，本轮没有重启或修改它；20:08观察已完成7/12配置，正在 Full-fixed-half（epoch16）。

| 目标 | 恢复数据集/seed | 断点已完成轮数 | 恢复后下一轮 | 剩余配置数（含中断项） | 等待进程PID |
|---|---|---:|---:|---:|---:|
| A GPU0 | SADD / 0 | 94/200 | 95 | 4 | 39856 |
| B GPU0 | SADD / 1 | 93/200 | 94 | 4 | 32614 |
| D GPU0 | SADD / 2 | 101/200 | 102 | 4 | 38286 |
| F GPU0 | SSDD / 0 | 31/150 | 32 | 10 | 41252 |
| F GPU1 | SSDD / 1 | 33/150 | 34 | 10 | 41253 |

PID仅是启动时身份；回执另存boot ID与进程start ticks。后续须同时检查实际命令、GPU UUID、queue_state，不按旧PID盲目操作。

五条队列合计承接32个尚未完成的配置训练及独立测试，配置顺序、种子、总epoch预算和超参数保持原矩阵。SSDD原已完成的Original/Matched-Identity分别保留原始结果路径，不重复训练。C的剩余配置不在这32项内。

## 队列顺序与安全条件

前序为各卡已经运行的 `codex_sar_agent_final_v1p1_20261008` 实验队列，各分片20项。登记时均为4/20完成、下一项训练中。只读取其调度状态，不修改其代码、状态或任务。

启动恢复训练须同时满足：

1. 前序 `frozen_queue_state.json` 的队列签名及GPU UUID与登记一致。
2. 状态是 `complete` / `complete_shard`，且全部预定seed/arm完成；前序调度进程已退出。
3. 对应GPU无计算进程、显存≤2500MiB、利用率≤10%，连续120秒成立；每30秒检查一次。
4. GPU UUID/编号、原冻结源码哈希仍一致；每次训练和测试前再次核查。

不会在前序两项训练的空隙插队。前序失败或未结束则继续等待；身份/源码变化、恢复子进程失败则失败停止，不自动改变超参数。恢复期间持有新队列锁及旧lane锁，防止原入口重复启动。

未安装软件，未修改服务或系统自启动，未中断其他任务，未在本机跑模型。后台进程脱离SSH/聊天运行，但**不具备服务器重启后的自动重启能力**；再次重启需核查断点后显式恢复。

## 独立目录与数据保护

各服务器恢复根：

`/home/powerleader/ig-ds-ssm/tasks/rgsp_recovery_20261008_v1/`

各lane为`sadd_seed0`、`sadd_seed1`、`sadd_seed2`、`ssdd_seed0`、`ssdd_seed1`。只在所属服务器存在相应lane。

- `recovery_manifest.json`：原实验路径、冻结源码/配置/数据哈希、断点SHA、前序队列签名、原结果来源。
- `verification.json`：CPU-only最终核验与原件哈希复核。
- `launch_receipt.json`：启动身份与等待确认。
- `queue_state.json`、`queue.log`：恢复队列的当前状态与诊断日志。
- `runs/`、`logs/`、`evaluations/`、`results_partial.json`：本次恢复的后续结果。

原两个campaign、全部已完成成绩和中断记录保持原样；新lane复制数据及中断run。所有复制图像/标签逐项核对原manifest。恢复使用原冻结业务代码，独立`train_redirect.py`只将resume的数据/输出路径转到新lane，不改变架构、优化器或训练预算。

旧F/A/B/D的`status.json=running`为重启前遗留值，不再代表实际运行。恢复状态以新根的`queue_state.json`与实际进程为准；旧C的队列仍有效。

## 核验与实验边界

- 现场复核A/B/C/D/F的主机、GPU UUID、进程、负载及磁盘，选择原A/B/D/F承接各自seed以减少执行条件变化。D通过已登记LAN别名连接；C不动。
- 五个last及best的ZIP CRC、CPU反序列化、模型有限值、seed/epoch/batch/优化器身份及控制常量检查通过。优化器张量有限值检查通过：SADD每lane1212项、SSDD每lane1167项。
- SSDD每lane248项、SADD每lane251项冻结代码哈希通过；原run文件共66项哈希复核未变。
- 三项门控/记录处理单元测试在A/B/D/F各通过；五个真实checkpoint的resume路径探针通过，确认输出路径不会回写原run。没有在忙碌GPU上做模型短训练。
- 首次远程测试赶在文件传输结束前返回“0 tests”；该次不计通过。传输全部完成、脚本SHA与本地一致后，在四台服务器重新执行，均明确运行3项测试通过。
- 重启前部分JSONL/CSV尾部为NUL；仅在新副本去除尾部NUL。CSV从checkpoint自身保存的`train_results`重建，补回A94、B93、F0-31、F1-33轮的真实记录；未改任何原始文件。
- 上述四轮的历史逐轮control审计缺失，保持缺口，不使用EMA值伪造；D无此缺口。后续重新产生真实逐轮审计。
- 旧checkpoint保存EMA及FP16优化器状态，不含完整RNG重放；本次是明确标注的resume，**不是逐位等价的不中断训练**。尚未执行恢复后的GPU训练，实际续训成功需后续`runtime_audit_resume.json`和新epoch/完成记录确认。
- 训练完成后仍按原验证选模与独立test协议评估，保留全部正负结果；不因恢复成功宣称消融或论文实验完成。

本地源脚本与直接证据：[恢复目录](../experiments/rgsp_recovery_20261008_v1/)。状态回写RunId：`rgsp-recovery-queued-20261008-01a11b5e`。
