# 文档验收 - 2026-10-09

- 主文7页，IEEE Letter双栏；补充材料8页，A4单栏。
- XeLaTeX多遍编译成功，最终日志无Overfull、缺失字符或未解析图表/文献引用。
- 主文保留IEEEtran/xeCJK的字体形状替代及少量Underfull警告；已通过PNG视觉检查，无可见缺字或裁切。
- 主文全部页与补充材料全部页已检查；图2为全矢量，图1含原始SAR裁块。
- 9张历史表的数据区域数值序列与输入版逐项一致；15项历史SSDD CSV未改；12项SADD按JSON复算均值和样本标准差。
- 统计和输入哈希详见evidence/DOCUMENT_AUDIT.json；包内文件逐项哈希详见PACKAGE_MANIFEST.json。
- 构建临时aux/log文件不纳入ZIP；最终ZIP解压后的复编译与PDF一致性回验记录单独保存于交付目录verification.json。
- 文档验收不等于完成全部实验或主文献数值核验。范围限制见SOURCE_REGISTRY.md。
