# Mamba-YOLO 项目管理规范（CODEX_RULES）

## 项目背景

本项目基于 Mamba-YOLO 开源项目进行长期研究与改进。

目前项目分散在多台服务器上，已经进行了大量实验。

可能存在：

* 多个代码版本；
* 多个实验分支；
* 多个配置版本；
* 不完整的实验统计表；
* 重复实验；
* 未整理的日志和权重。

本地目录当前为空。

需要建立一个统一、完整、可追溯的主仓库。

---

# 一、最高目标

恢复并建立以下映射关系：

实验名称
→ 数据集
→ 配置文件
→ 代码版本
→ Git Commit
→ 模型结构
→ 实验结果

最终能够明确回答：

某个实验结果是由哪份代码、哪份配置、在哪台服务器上运行得到的。

该目标优先级高于代码整理。

---

# 二、本地主仓库

本地工作目录：

D:\A-主机文件\Codex\Mamba-YOLO-ds-ssm

当前为空目录。

未来将作为：

Mamba-YOLO Master Repository

（唯一可信主仓库）。

不要假设本地存在正确代码。

正确代码可能位于任意服务器。

在完成服务器盘点和代码差异分析之前：

禁止认定任何服务器上的代码为最终版本。

---

# 三、本地目录限制

只允许操作：

D:\A-主机文件\Codex\Mamba-YOLO-ds-ssm

及其子目录。

禁止：

* 访问该目录之外的文件；
* 搜索该目录之外的文件；
* 修改该目录之外的文件；
* 创建该目录之外的文件；
* 删除该目录之外的文件；

即使系统允许访问，也禁止操作。

---

# 四、服务器访问规则

服务器仅作为历史实验数据源。

服务器必须视为只读。

允许：

* ssh
* ls
* find
* grep
* cat
* head
* tail
* tree
* du
* git status
* git log
* git diff
* 查看日志
* 查看配置
* 查看权重
* 下载文件
* 下载目录
* rsync 下载
* scp 下载

禁止：

* rm
* mv
* cp 覆盖
* sed -i
* vim
* nano
* 修改代码
* 修改配置
* 修改日志
* 删除文件
* 删除实验
* 删除权重
* git commit
* git push
* git checkout
* git reset
* git clean
* pip uninstall
* conda remove

任何可能改变服务器状态的操作均禁止。

---

# 五、服务器信息

## Server-1080Ti

IP：
172.26.193.188

用户：
zhangji

密码：
123456

项目目录：
/home/zhangji/Mamba-YOLO

实验目录：
/home/zhangji/Mamba-YOLO/output_dir

GPU：
2 × 1080Ti

---

## Server-3090-A

IP：
172.26.150.36

用户：
zhangji

密码：
12345

项目目录：
/home/zhangji/Mamba-YOLO

实验目录：
/home/zhangji/Mamba-YOLO/output_dir

GPU：
1 × RTX 3090

---

## Server-3090-B

IP：
172.26.133.16

用户：
zhangji

密码：
12345

项目目录：
/home/zhangji/Mamba-YOLO

实验目录：
/home/zhangji/Mamba-YOLO/output_dir

GPU：
1 × RTX 3090

---

## 跳板机

IP：
172.26.151.143

用户：
zhangji

如访问目标服务器需要跳板机，则自动使用。

---

# 六、服务器目录限制

允许访问：

/home/zhangji/Mamba-YOLO

及其子目录。

禁止访问：

/root

/etc

/var

/home 下其它用户目录

以及与项目无关的目录。

---

# 七、阶段0：服务器真实性确认

目标：

确认三台服务器之间的关系。

检查：

* Git 仓库状态
* Branch
* Commit 数量
* 最新 Commit 时间
* 未提交修改数量
* output_dir 实验数量
* 权重数量
* 日志数量

判断：

1. 是否为同一项目分支
2. 是否存在主开发服务器
3. 是否存在长期未同步服务器
4. 是否存在仅用于训练的服务器

输出：

inventory/server_relationship.md

在完成分析前：

禁止认定任何服务器为最终版本。

---

# 八、官方项目对比原则

必须以官方 Mamba-YOLO 项目为基准。

不要仅关注用户提到的文件。

识别：

* 新增文件
* 删除文件
* 修改文件
* 新增模块
* 修改模块
* 修改训练流程
* 修改验证流程
* 修改推理流程
* 修改数据处理流程
* 修改损失函数
* 修改模型结构
* 修改配置文件

输出：

reports/project_diff_report.md

---

# 九、高优先级分析目录

优先分析：

ultralytics/

ultralytics/nn/

ultralytics/nn/modules/

ultralytics/models/

ultralytics/engine/

ultralytics/data/

ultralytics/utils/

cfg/

configs/

tools/

scripts/

以及项目根目录所有：

*.py

*.yaml

*.yml

文件。

---

# 十、第一阶段：项目盘点

首次运行时：

只允许执行项目盘点。

禁止：

* 下载全部代码
* 下载全部实验
* 下载大量权重
* 自动合并代码
* 自动选择主版本

收集：

1. git status
2. 当前分支
3. commit 历史
4. output_dir 实验数量
5. 权重数量
6. 日志数量
7. 配置文件数量
8. 三台服务器代码是否一致
9. 哪些文件存在差异
10. 与官方仓库差异情况

生成：

inventory/server_1080ti.md

inventory/server_3090_a.md

inventory/server_3090_b.md

inventory/server_relationship.md

inventory/project_overview.md

inventory/code_diff_overview.md

完成后停止并等待确认。

---

# 十一、第二阶段：实验恢复

实验恢复优先目录：

/home/zhangji/Mamba-YOLO/output_dir

优先扫描：

best.pt

last.pt

results.csv

results.png

args.yaml

opt.yaml

train.log

events.out.tfevents*

以及其它实验记录文件。

如果 output_dir 信息完整：

不要扩大扫描范围。

---

# 十二、实验数据库

建立：

database/experiment_summary.xlsx

database/experiment_summary.csv

database/experiment_database.json

每个实验记录：

* 实验名称
* 来源服务器
* 实验目录
* 时间
* 数据集
* 配置文件
* 训练命令
* Epoch
* Batch Size
* Image Size
* 学习率
* FLOPs
* Params
* Precision
* Recall
* mAP50
* mAP50-95
* best.pt
* last.pt

---

# 十三、代码版本恢复

建立：

实验
→ 配置
→ 代码版本
→ Git Commit
→ 结果

映射关系。

生成：

reports/code_version_map.md

---

# 十四、代码差异矩阵

如果发现多个服务器存在不同代码版本：

禁止自动合并。

必须先生成：

reports/code_diff_matrix.md

内容包括：

* 文件路径
* 服务器来源
* 修改时间
* Git Commit
* 差异摘要
* 关联实验

等待用户确认。

---

# 十五、实验去重

识别：

* 重复实验
* 重复配置
* 重复权重
* 重复日志
* 重复结果

生成：

reports/duplicate_report.md

禁止自动删除。

---

# 十六、安全规则

以下操作必须获得确认：

* 删除文件
* 覆盖文件
* 重命名文件
* 合并文件
* 调整目录结构
* 清理日志
* 清理权重

未经确认禁止执行。

---

# 十七、工作流程

严格遵循：

分析
→ 生成报告
→ 等待确认
→ 执行

任何时候：

优先保护数据。

优先保证实验可追溯性。

优先恢复实验与代码的映射关系。
