# 证据范围

- NUMERICAL_AUDIT.json：正文所用SSDD15run与SADD12run统计，比例乘100，样本标准差ddof=1，同seed配对，不排除不利seed。
- DOCUMENT_VALIDATION.json：本轮新编译PDF的pypdf运行核验PASS，包含27项原件、九张历史表、逐seed指标、统计/配对、页数及13项输入哈希；不是模型测试报告。
- DOCUMENT_VALIDATION_POPPLER.json：同一新PDF由现有TeX Live的pdfinfo/pdftotext完成的独立运行核验PASS；不同文本后端不要求提取文本哈希相同。
- BUILD_VALIDATION.json：用户授权本机文档回退后的首轮独立解压、双遍XeLaTeX编译回执；记录新PDF/输入/最终日志哈希、文本与14页渲染一致性。最终ZIP往返运行回执另见交付目录verification.json。
- STATIC_DOCUMENT_VALIDATION_20261009.json：本机回退获准前的静态验收历史记录；其中未运行/未授权字段仅属于当时状态，不是当前结论。
- PREEXISTING_BUILD.json：续接前旧14页PDF的既有编译依据，原样保留历史；不是本轮新PDF或最终ZIP的编译回执。
- COMPILE_INPUTS.json：主文、全部输入章节、IEEEtran类和两张PDF图的交付前哈希；不包含随包Markdown说明或核验脚本。
- frozen_ssdd与frozen_sadd：此前已回收冻结代码中的模块副本，mamba_yolo.py SHA256均为c5248810985a825e32eeef4b276b821230ce1b999970e73ee9d87ecb8f5a7add。用于对应本文图件和既有消融定义，不自动替换当前业务源码。
- recovery：此前回收的恢复核验摘要，不能将其本身当作最终完成回执。SADD完成来源为2026-10-09回收JSON及项目reports/SADD_RESULTS_SSDD_RESTART_20261009.md。

服务器传输失败和旧静态包均保留为历史；随后获得的本机授权仅用于文档编译、核验和打包。本轮没有实验队列核查、训练、推理或自定义CUDA执行，也没有软件安装或服务改动。其他并发任务对业务代码、实验队列的改变不属于本轮。
