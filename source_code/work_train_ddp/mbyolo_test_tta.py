from ultralytics import YOLO
import argparse
import os

ROOT = os.path.abspath('.') + "/"


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT + '/ultralytics/cfg/datasets/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--config', type=str, default=ROOT + '/ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml', help='model path(s)')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--task', default='train', help='train, val, test, speed or study')
    parser.add_argument('--device', default='0,1,2,3,4,5,6,7', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=128, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--optimizer', default='SGD', help='SGD, Adam, AdamW')
    parser.add_argument('--amp', action='store_true', help='open amp')
    parser.add_argument('--project', default=ROOT + '/output_dir/mscoco', help='save to project/name')
    parser.add_argument('--name', default='mambayolo', help='save to project/name')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    # 🚀 新增：测试阶段专属打榜参数 (TTA, NMS, Conf)
    parser.add_argument('--augment', action='store_true', help='open Test-Time Augmentation (TTA)')
    parser.add_argument('--iou', type=float, default=0.7, help='NMS IoU threshold for validation')
    parser.add_argument('--conf', type=float, default=0.001, help='confidence threshold for validation')
    parser.add_argument('--max_det', type=int, default=300, help='maximum detections per image')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    task = opt.task
    args = {
        "data": ROOT + opt.data,
        "epochs": opt.epochs,
        "imgsz": opt.imgsz,
        "workers": opt.workers,
        "batch": opt.batch_size,
        "optimizer": opt.optimizer,
        "device": opt.device,
        "amp": opt.amp,
        "project": ROOT + opt.project,
        "name": opt.name,
    }


    # model_conf = ROOT + opt.config
    # model = YOLO(model_conf)

    # if task == "train":
    #     model.train(**args)
    # elif task == "val":
    #     model.val(**args)
    # elif task == "test":
    #     model.val(**args)
    # else:
    #     raise ValueError(f"Unsupported task: {task}")

    # elif task == "val" or task == "test":
    #     print("\n🚀 [战术执行] 注入极限打榜测试参数 (放宽 NMS，拉低 Conf)...")
    #     # 复制原本的 args
    #     test_args = args.copy()
    #     # 强行更新后处理参数
    #     test_args.update({
    #         "iou": 0.6,        # 放宽 NMS，避免密集小船被互相抑制误杀
    #         "conf": 0.001,     # 拉低置信度下限，把所有疑似目标都算入召回率
    #         "max_det": 1000    # 确保极端密集图不会因为数量上限而被截断
    #     })
    #     # 传入带有作弊参数的字典进行测试
    #     model.val(**test_args)
    
    
    model_conf = ROOT + opt.config
    model = YOLO(model_conf)

    if task == "train":
        # 训练时保持原样
        model.train(**args)
        
    elif task == "val" or task == "test":
        # 🟢 测试时，把新加的打榜参数合并进去
        test_args = args.copy()
        test_args.update({
            "augment": opt.augment,   # 核心：触发测试时多尺度/翻转数据增强
            "iou": opt.iou,           # 控制框的重叠抑制
            "conf": opt.conf,         # 控制预测的底线置信度
            "max_det": opt.max_det    # 确保密集小目标不被截断
        })
        print(f"\n🚀 [测试启动] 注入战术参数: Augment={opt.augment}, IoU={opt.iou}, Conf={opt.conf}")
        model.val(**test_args)
        
    else:
        raise ValueError(f"Unsupported task: {task}")
