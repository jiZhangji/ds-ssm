# SADD结果与SSDD恢复（2026-10-09，北京时间）

用户授权查看SADD、暂停ComfyUI并禁止自动启动、恢复SSDD训练。

## SADD结果

四配置×三个种子，12个正式训练和test已完成。2026-10-09在A/B/D逐项检查train_complete、eval_test、evaluation_complete及best.pt SHA256，均PASS；原始指标回收于experiments/rgsp_status_20261009/sadd_seed0/1/2.json。

AP50:95，单位%，均值±样本标准差：

|配置|seed0|seed1|seed2|均值±标准差|
|---|---:|---:|---:|---:|
|Original|53.7117|53.9804|52.3480|53.3467±0.8753|
|Matched075|52.0914|53.0962|51.3444|52.1773±0.8791|
|LocalEnergy|51.9479|52.3537|52.8014|52.3676±0.4269|
|Full|49.0030|53.1989|53.7361|51.9793±2.5915|

Full−Original三seed配对差值−4.7087/−0.7816/+1.3881pp，平均−1.3674pp，仅1/3正差。LocalEnergy−Matched075均值+0.1903pp；Full−LocalEnergy均值−0.3883pp，2/3正差但seed0明显负差。不能宣称完整方法在SADD稳定提升，三seed不足以作显著性断言。

边界：positive-only、固定795/44/44、test44；不是已证明场景独立协议。Original从EMA/FP16优化器断点恢复，非逐位无中断等价；A/B历史审计缺口保留。未删除负结果，未改论文。

## ComfyUI

C用户systemd transient comfyui.service已处于failed，无现存ComfyUI计算进程；本轮stop/reset-failed，再持久mask。核验is-enabled=masked、is-active=inactive，~/.config/systemd/user/comfyui.service指向/dev/null。未发现该用户cron、systemd/autostart及所查系统入口中的其他ComfyUI自启动项。mask阻止该unit自动启动；未来如人为使用其他unit名或直接脚本启动，仍需单独管理。

## SSDD恢复

- C原seed2：保存旧status后重新执行原冻结队列，已跳过完成训练并补齐guidance_only test，11/12；global新训练已确认epoch8有限损失。调度PID3594308、训练PID3594331仅为时点身份。
- F seed0/1：失败原因明确为train_redirect传入冻结SSDD CLI不支持的--epochs/--patience/--batch，退出码2，未进入续训。包装器静态识别CLI支持；SSDD仍由冻结trainer内部固定150/30/16，SADD保留显式参数。冻结业务代码未修改。
- F备份旧包装器、manifest和失败状态，更新包装器SHA并复查原断点/冻结代码/GPU身份及空闲；重启原有限恢复队列PID122224/122228，各剩10配置，仍执行前序完成与连续120秒空闲门控。
- 恢复包装器SHA256：3afdc6f6ce4099836d837969b253ff763dc07fb5101dd2dc599ebf4b03bb2920。源及重启脚本在experiments/rgsp_recovery_20261008_v1。连接不稳定时使用已登记B作为跳板访问已登记F/D LAN地址，未扫描网段。

服务器盘点A/B/C/D/F：GPU初检均无计算进程；空闲磁盘约249/625/794/144/217GB；CPU低负载。选择原C/F原卡以保持执行条件，未本机运行模型、安装软件或抢占任务。
`n最终复核：F0/F1已通过门控，Matched075新epoch32/34记录有限损失，runtime_audit_resume.json均存在，GPU95%/100%；确认真实续训。C Global已确认训练推进。正式记忆发布回执COMMITTED。
