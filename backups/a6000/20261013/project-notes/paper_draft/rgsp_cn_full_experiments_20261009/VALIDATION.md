# 交付验收状态 - 2026-10-09

状态：本地文档编译与核验PASS。最终ZIP的运行验收回执为交付目录verification.json；该回执在完成独立解压、双遍XeLaTeX编译和全部14页一致性核验后才写入PASS，不能用旧verification_static.json替代。这是文档交付验收，不代表新增研究实验完成。

## 文档核验

- 14页IEEE双栏正文，18张表、2张图。原九张分类/检测/分割/变化检测任务表保留在正文，消融包含SSDD历史15run、SADD新12run的135个逐种子指标，9组均值/样本标准差、12种配对比较及12配置控制定义。
- 逐格静态复核未发现数字或归因错误，所有负结果与实验限制保留。全部14页既有渲染图已逐页审阅，未发现裁切、重叠、缺字或不可读表格；新PDF在115dpi下与该已审阅渲染集的14张PNG逐页同哈希。第12-13页为正常双栏浮动表页。
- 新编译PDF SHA256：`3ae0914cbd50e779cd1cd53569c16152d780fefe926eb09fec66bd08a81a00bd`。旧PDF SHA256为`6ee5f38a6fb1af0d1f98d23ca35fb14c011db23cbe0290f77817f17f39871cfb`；二进制不同不等于正文变化。两者全部逐页文本、页面尺寸及渲染PNG相同，pypdf全文哈希均为`e0f9934c92c8c7ab5a8a288652277d2e1c9a454a2664d1b1a838d667890db524`。
- 新编译输入来自上一版静态ZIP的独立解压，83项payload哈希全部匹配。main.tex、9个输入章节、IEEEtran.cls及两张图共13项编译输入未变，见evidence/COMPILE_INPUTS.json。
- 当前tools/verify_document.py已分别通过pypdf与既有pdfinfo/pdftotext两种后端的实际运行：原上传ZIP27项原件、九张历史表数值序列、15+12run、135个指标、统计与配对打印值、18表/2图、14页612×792pt及13项编译输入均通过。两种文本引擎的提取布局不同，不要求跨引擎文本哈希相同。
- 第二遍编译日志没有Overfull、缺字、未解析引用或要求重跑的警告；既有字体形状替代与参考文献Underfull警告保留。

## 最终ZIP往返验收

tools/package_delivery.py要求BUILD_VALIDATION、DOCUMENT_VALIDATION和13项输入清单绑定同一PDF。它仅创建新输出，不覆盖上一版ZIP；先核对全部ZIP成员、CRC、SHA256及解压内容，再从独立解压目录执行XeLaTeX两遍，并核验打包PDF与重编译PDF的逐页文本、尺寸、输入和115dpi的14张PNG全部一致。成功回执要求status=PASS及roundtrip_compilation.status=PASS；任一步失败不产生成功回执。

最终验证报告位于output/pdf/rgsp_full_experiments_verified_20261009/verification.json。源码包中的SHA256SUMS.json覆盖所有payload但不包含自身，避免循环哈希；最终ZIP哈希写在包外回执中。编译日志、两种PDF的渲染图保留在本轮独立临时目录供复查。

## 执行目标与授权

服务器优先盘点曾检查A/B/C/D/F：初检均可SSH，B忙；常见目录与PATH未找到现成TeX。19:38复检A已有GPU任务，改选低负载F，但独立目录创建后的SCP以及后续A/C/D/F和已配置LAN路径超时，远端文档核验未执行。远端可能保留不完整临时副本，不作完整备份使用。

用户随后明确同意仅对文档采用本机回退。本轮使用现有TeX Live 2026、Python 3.12.14/pypdf 6.10.0和Poppler渲染工具，无安装。内置Poppler目录缺少pdftotext.exe，已使用现有TeX Live的pdfinfo.exe/pdftotext.exe完成备用文本核验；Windows pdfinfo日期的本地编码通过仅按字节解析ASCII页数和MediaBox解决，不放宽PDF正文UTF-8检查。没有训练、推理、模型测试、服务启停或实验队列核查/改动。

## 仍未完成的研究事项

新SSDD完整36run、统一效率、机制诊断、公平外部基线与外部表数值主文献核验均不因本次文档验收而完成。旧英文稿P2/4冲突也不在本次关闭范围。

可复现入口：tools/verify_document.py需要pypdf或既有pdfinfo/pdftotext；tools/package_delivery.py还需要已有XeLaTeX和pdftoppm，并默认使用pypdf核验。运行时显式提供新的--zip、--report、--extract-to和--compile-to路径，不覆盖历史交付。
