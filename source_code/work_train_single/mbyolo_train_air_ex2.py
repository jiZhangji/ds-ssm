# from ultralytics import YOLO
# import argparse
# import os

# ROOT = os.path.abspath('.') + "/"


# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data', type=str, default=ROOT + '/ultralytics/cfg/datasets/coco.yaml', help='dataset.yaml path')
#     parser.add_argument('--config', type=str, default=ROOT + '/ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml', help='model path(s)')
#     parser.add_argument('--batch_size', type=int, default=512, help='batch size')
#     parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
#     parser.add_argument('--task', default='train', help='train, val, test, speed or study')
#     parser.add_argument('--device', default='0,1,2,3,4,5,6,7', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--workers', type=int, default=128, help='max dataloader workers (per RANK in DDP mode)')
#     parser.add_argument('--epochs', type=int, default=300)
#     parser.add_argument('--optimizer', default='SGD', help='SGD, Adam, AdamW')
#     parser.add_argument('--amp', action='store_true', help='open amp')
#     parser.add_argument('--project', default=ROOT + '/output_dir/mscoco', help='save to project/name')
#     parser.add_argument('--name', default='mambayolo', help='save to project/name')
#     parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
#     parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
#     opt = parser.parse_args()
#     return opt


# if __name__ == '__main__':
#     opt = parse_opt()
#     task = opt.task
#     args = {
#         "data": ROOT + opt.data,
#         "epochs": opt.epochs,
#         "imgsz": opt.imgsz,
#         "workers": opt.workers,
#         "batch": opt.batch_size,
#         "optimizer": opt.optimizer,
#         "device": opt.device,
#         "amp": opt.amp,
#         "project": ROOT + opt.project,
#         "name": opt.name,
#     }
#     # model_conf = ROOT + opt.config
#     # task_type = {
#     #     "train": YOLO(model_conf).train(**args),
#     #     "val": YOLO(model_conf).val(**args),
#     #     "test": YOLO(model_conf).test(**args),
#     # }
#     # task_type.get(task)

#     # model_conf = ROOT + opt.config
#     # task_type = {
#     #     "train": YOLO(model_conf).train(**args),
#     #     "val": YOLO(model_conf).val(**args),
#     #     "test": YOLO(model_conf).val(**args),
#     # }
#     # task_type.get(task)


#     model_conf = ROOT + opt.config
#     model = YOLO(model_conf)

#     if task == "train":
#         model.train(**args)
#     # elif task == "val":
#     #     model.val(**args)
#     # elif task == "test":
#     #     model.val(**args)
#     # else:
#     #     raise ValueError(f"Unsupported task: {task}")

#     elif task == "val" or task == "test":
#         print("\n🚀 [战术执行] 注入极限打榜测试参数 (放宽 NMS，拉低 Conf)...")
#         # 复制原本的 args
#         test_args = args.copy()
#         # 强行更新后处理参数
#         test_args.update({
#             "iou": 0.6,        # 放宽 NMS，避免密集小船被互相抑制误杀
#             "conf": 0.001,     # 拉低置信度下限，把所有疑似目标都算入召回率
#             "max_det": 1000    # 确保极端密集图不会因为数量上限而被截断
#         })
#         # 传入带有作弊参数的字典进行测试
#         model.val(**test_args)


from ultralytics import YOLO
import argparse
import os

ROOT = os.path.abspath('.') + "/"

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT + 'ultralytics/cfg/datasets/SAR-AIRcraft1.0.yaml', help='dataset.yaml path') # 建议换成航空器数据集
    parser.add_argument('--config', type=str, default=ROOT + 'ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml', help='model path(s)')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size') # 航空器大图建议调小 batch
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=800, help='inference size (pixels)') # 航空器强烈建议 800 或 1024
    parser.add_argument('--task', default='train', help='train, val, test, speed or study')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--optimizer', default='SGD', help='SGD, Adam, AdamW')
    parser.add_argument('--amp', action='store_true', help='open amp')
    parser.add_argument('--project', default=ROOT + 'output_dir/sar_aircraft', help='save to project/name')
    parser.add_argument('--name', default='mambayolo_ds_ssm', help='save to project/name')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    return opt


# 🔴 新增：定义两阶段动态冻结的回调函数
# def dynamic_freeze_callback(trainer):
#     freeze_epochs = 20  # 设定前 20 轮只训练去噪参数
    
#     if trainer.epoch == 0:
#         print(f"\n❄️ [战术执行] 启动冻结热身！冻结主干老权重，专心训练去噪模块 (前 {freeze_epochs} 轮)...")
#         for name, param in trainer.model.named_parameters():
#             # 只要参数名字里包含咱们新写的去噪变量，就放行 (让它们去学习)
#             if any(keyword in name for keyword in ['lambda_s', 'alpha', 'damping_bias', 'dw']):
#                 param.requires_grad = True
#                 # print(f"  -> [训练中] {name}") # 如果你想确认哪些层在练，可以取消这行注释
#             else:
#                 param.requires_grad = False
                
#     elif trainer.epoch == freeze_epochs:
#         print(f"\n🔥 [战术执行] 热身完毕 (Epoch {trainer.epoch})！全网解冻，开启全局联合微调...")
#         for name, param in trainer.model.named_parameters():
#             param.requires_grad = True


# # 冻结滤波对应的参数。  ex14-ex15.
# def dynamic_freeze_callback(trainer):
#     freeze_epochs = 20  # 设定前 20 轮只训练去噪参数
    
#     if trainer.epoch == 0:
#         print(f"\n❄️ [战术执行] 启动冻结热身！冻结主干老权重，专心训练去噪模块 (前 {freeze_epochs} 轮)...")
#         for name, param in trainer.model.named_parameters():
#             # 🔴 关键修复：加入了 'homo_scale' 和 'homo_filter'
#             whitelist = ['lambda_s', 'alpha', 'damping_bias', 'dw', 'homo_scale', 'homo_filter']
            
#             if any(keyword in name for keyword in whitelist):
#                 param.requires_grad = True
#                 # 💡 强烈建议你把下面这行打印取消注释，跑的时候看一眼屏幕，确认它们真的在训练！
#                 print(f"  -> [保持训练的参数] {name}") 
#             else:
#                 param.requires_grad = False
                
#     elif trainer.epoch == freeze_epochs:
#         print(f"\n🔥 [战术执行] 热身完毕 (Epoch {trainer.epoch})！全网解冻，开启全局联合微调...")
#         for name, param in trainer.model.named_parameters():
#             param.requires_grad = True


#   ex16-gemini  -ex22
# def dynamic_freeze_callback(trainer):
#     freeze_epochs = 20  # 设定前 20 轮只训练去噪参数
    
#     if trainer.epoch == 0:
#         print(f"\n❄️ [战术执行] 启动冻结热身！冻结主干老权重，专心训练去噪模块 (前 {freeze_epochs} 轮)...")
#         for name, param in trainer.model.named_parameters():
            
#             # 🔴 最新白名单：加入了 homo_scale 和 gated_filter
#             whitelist = ['lambda_s', 'alpha', 'damping_bias', 'dw', 'homo_scale', 'gated_filter']
            
#             if any(keyword in name for keyword in whitelist):
#                 param.requires_grad = True
#                 # 打印出来求个心安，确保它们真正在训练
#                 print(f"  -> [保持训练的去噪/滤波参数] {name}") 
#             else:
#                 param.requires_grad = False
                
#     elif trainer.epoch == freeze_epochs:
#         print(f"\n🔥 [战术执行] 热身完毕 (Epoch {trainer.epoch})！全网解冻，开启全局联合微调...")
#         for name, param in trainer.model.named_parameters():
#             param.requires_grad = True

# gemini-ex23
def dynamic_freeze_callback(trainer):
    freeze_epochs = 20  # 前 20 轮热身
    
    if trainer.epoch == 0:
        print(f"\n❄️ [战术执行] 启动冻结热身！冻结主干老权重，专心训练去噪/感知模块 (前 {freeze_epochs} 轮)...")
        for name, param in trainer.model.named_parameters():
            
            # 🔴 精确更新：加入了 dw2d (局部估计) 和 mlp_dwconv (Conv-FFN 的核心)
            whitelist = ['lambda_s', 'alpha', 'damping_bias', 'dw2d', 'homo_scale', 'gated_filter', 'mlp_dwconv']
            
            if any(keyword in name for keyword in whitelist):
                param.requires_grad = True
                print(f"  -> [保持训练的新参数] {name}") 
            else:
                param.requires_grad = False
                
    elif trainer.epoch == freeze_epochs:
        print(f"\n🔥 [战术执行] 热身完毕 (Epoch {trainer.epoch})！全网解冻，开启全局联合微调...")
        for name, param in trainer.model.named_parameters():
            param.requires_grad = True


if __name__ == '__main__':
    opt = parse_opt()
    task = opt.task
    
    # 基础参数配置
    args = {
        "data": opt.data, # 注意：你原本的 default 里已经加了 ROOT，这里就不要重复加 ROOT + opt.data 了
        "epochs": opt.epochs,
        "imgsz": opt.imgsz,
        "workers": opt.workers,
        "batch": opt.batch_size,
        "optimizer": opt.optimizer,
        "device": opt.device,
        "amp": opt.amp,
        "project": opt.project,
        "name": opt.name,
    }
    
    # # 🔴 新增：自动注入 ex9 验证过的 SAR 最佳物理超参数，免去每次在命令行敲打的麻烦
    # sar_specific_args = {
    #     "box": 2.0,      # 给框回归松绑
    #     "cls": 3.0,      # 死保分类，消灭虚警
    #     "dfl": 0.0,      # 彻底放弃模糊边缘执念
    #     "hsv_h": 0.0,    # 关闭光学色相增强
    #     "hsv_s": 0.0,    # 关闭光学饱和度增强
    #     "hsv_v": 0.0     # 关闭光学明度增强
    # }
    # args.update(sar_specific_args)

    model_conf = ROOT + opt.config
    
    # 1. 初始化模型架构
    model = YOLO(model_conf)
    
    # 🔴 关键步骤：加载官方光学预训练权重作为底子
    # 请确保你的工程目录里有这个文件，或者改成你本地对应的预训练权重名
    try:
        model.load("yolov8n.pt") # Mamba-YOLO 通常也兼容/基于对应的 YOLOv8 预训练权重体系
        print("✅ 成功加载预训练权重底子！")
    except Exception as e:
        print(f"⚠️ 警告: 未能加载预训练权重，模型将从头训练 ({e})")

    # 2. 执行任务
    if task == "train":
        # 🔴 新增：在正式 train 之前，把我们的冻结战术作为“插件”挂载上去
        model.add_callback("on_train_epoch_start", dynamic_freeze_callback)
        model.train(**args)
        
    elif task == "val":
        model.val(**args)
    elif task == "test":
        model.val(**args)
    else:
        raise ValueError(f"Unsupported task: {task}")