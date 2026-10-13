# SSDD 参数量与计算量核查（2026-10-11）

本轮用户授权先检索、缺失时在服务器测试，并明确允许建立隔离环境安装依赖。
检查两台项目3090服务器及A6000-D/F；3090忙，D/F空闲，选择F（更多磁盘余量）。
远端独立目录：`/home/powerleader/codex_tasks/complexity_20261011`。
使用env及env_legacy隔离目录，现有训练环境未修改；未训练、未评估检测精度。

## 实测结果

统一单类、FP32、输入1×3×640×640、乘加计2 FLOPs。参数为完整注册模型总参数。
THOP与fvcore支持的运算计数仍有覆盖差异，不视为包含全部算子的精确总运算量。

| 实现 | Params(M) | 报告GFLOPs | 说明 |
|---|---:|---:|---|
| YOLOv3 | 61.523734 | 155.259699 | Ultralytics YOLOv5 v7.0 hub实现 |
| YOLOv5n | 1.765270 | 4.224205 | 官方YOLOv5 v7.0 |
| YOLOv5s | 7.022326 | 15.945728 | 官方YOLOv5 v7.0 |
| YOLOv8n | 3.011043 | 8.194150 | 项目冻结Ultralytics实现 |
| FCOS R50-FPN | 32.113227 | 157.158477 | MMDetection3.3.0 |
| Cascade R-CNN R50-FPN | 69.151841 | 237.394893 | MMDetection3.3.0，随机初始化产生的proposal路径；ROIAlign/NMS未计 |
| FGBG-Net作者R50配置 | 32.711764 | 171.142362 | 作者master中的vfnet_r50_fpn_1x_coco.py；不是论文成绩版本绑定确认 |
| Original-clean | 5.985547 | 13.611565 | 冻结v3源码和对应Original配置，scan未计 |
| RGSP-clean | 5.920441 | 13.148179 | 冻结v3源码和对应Full配置，scan未计 |
| 历史ex7结构参考 | 5.964757 | 13.207853 | 不能未经检查点绑定填入ex27/P3所有历史行 |

项目现代Ultralytics v5配置实测2.508659M/7.176704G（n）、9.122579M/24.043059G（s），
与官方v7.0模型不是同一架构；主表采用官方v7.0参考值，并保留原始JSON。
现代项目YOLOv3为103.693235M/282.980403G，同样不可冒充经典YOLOv3。

## 已测但无法绑定原表规模

| 模型 | Params(M) | 报告GFLOPs |
|---|---:|---:|
| CenterNet HG52 | 104.783243 | 927.953306 |
| CenterNet HG104 | 209.941142 | 1302.507520 |
| YOLOX nano | 0.896754 | 2.548634 |
| YOLOX tiny | 5.032866 | 15.231130 |
| YOLOX s | 8.937682 | 26.756915 |
| YOLOX m | 25.280754 | 73.725850 |
| YOLOX l | 54.147986 | 155.671347 |
| YOLOX x | 98.995378 | 282.030592 |

CenterNet为Duankaiwen/CenterNet的Keypoint Triplets实现，非Objects as Points。
角点池化用等价方向cummax代替旧Python3.6扩展，解码移除以统计网络前向；池化/解码运算未计。
单类输出头仅将原80类别改为1。MaDiNet检测结果行未说明HG版本或YOLOX规模，主表不任意选择值。

## 修复、诊断与未完成项

- FGBG-Net仓库缺少19个MMDetection框架Python文件；从匹配的v2.25.2补齐任务副本。
  删除作者检测头中两个未使用的图像指标import，不改层或forward；保留修复列表JSON。
  DeformConv的卷积乘加已用fvcore自定义handler计入，插值等额外操作未计。
  旧151.029G漏计10次DeformConv，已撤回，最终为171.142G。
- FEPS-Net作者仅公开neck及调用外部DCN依赖。本轮按RetinaNet R50、stride4--64和兼容DCN重建诊断，
  得到37.538241M/485.488323G；这不是已确认的原论文完整配置，不填入原方法主表。
  初始466.614G遗漏DCN卷积，最终诊断已补卷积计数。必须先获得原完整配置/DCN依赖才能绑定。
- PEGNet（DOI10.1109/LGRS.2025.3528221）未找到确认的作者完整代码，全文访问受限。
  同名图网络/区块链项目与该论文无关，未替代实现或编造数值。
- MaDiNet作者master重新下载核查，仍未提供可确认的完整扩散检测器配置/集成。
  既有223.2648M/345.283G只属于res4+res5核心模块，不作为完整模型。
- Faster R-CNN、RetinaNet、DiffusionDet、ConsistencyDet、DiffDet4SAR复用20260724已有测量，
  未重复测；其配置与文献精度版本绑定仍未确认。ConsistencyDet记录实际40次head调用。
- FLOPs主表注说明scan/ROIAlign等覆盖限制。未将FLOPs解释为实测速度。

## 产物与验证

- `artifacts/complexity_20261011/*results.json`：实际回收原始结果，含未计算子。
- `complexity_benchmark/*20261011.py`：本轮测量、源码下载和表格更新脚本。
- `artifacts/complexity_20261011/SSDD_TABLE_WITH_COMPLEXITY.tex`：完整可复制表格。
- 当前完整论文源码仅替换SSDD表：26/26方法行和原检测数值保持，外部按年排序，减少分组横线。
- 静态证据`STATIC_CHECKS.json`，未编译或验收新10列表格分页。
- 正式项目记忆尚未合并；保存独立增量。
