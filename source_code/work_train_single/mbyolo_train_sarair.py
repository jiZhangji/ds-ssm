from ultralytics import YOLO
import argparse
import os

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/"


def resolve_path(path):
    """支持相对路径和绝对路径，避免 ROOT 重复拼接"""
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT, path)


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data',
        type=str,
        default='ultralytics/cfg/datasets/SAR_AIRcraft.yaml',
        help='dataset.yaml path'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml',
        help='model yaml or weight path'
    )

    parser.add_argument(
        '--weights',
        type=str,
        default='yolov8n.pt',
        help="pretrained weights loaded into YAML models; use 'none' to train from scratch"
    )

    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=1024, help='image size')
    parser.add_argument('--task', default='train', help='train, val, test')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='dataloader workers')
    parser.add_argument('--epochs', type=int, default=300)

    parser.add_argument('--optimizer', default='AdamW', help='SGD, Adam, AdamW')
    parser.add_argument('--lr0', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='final learning rate ratio')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--patience', type=int, default=80)
    parser.add_argument(
        '--module_warmup_epochs',
        type=int,
        default=20,
        help='optimize only DS-SSM/PFIM module parameters for the first N epochs; use 0 to disable'
    )

    parser.add_argument('--amp', action='store_true', help='open amp')
    parser.add_argument('--cache', action='store_true', help='cache images for faster training')
    parser.add_argument('--cos_lr', action='store_true', help='use cosine lr scheduler')

    parser.add_argument('--project', default='output_dir/SAR_AIRcraft', help='save to project/name')
    parser.add_argument('--name', default='mambayolo_sarair', help='save to project/name')

    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')

    # val / test 参数
    parser.add_argument('--conf', type=float, default=0.001, help='confidence threshold for val/test')
    parser.add_argument('--iou', type=float, default=0.6, help='NMS IoU threshold for val/test')

    # SAR 数据增强参数
    parser.add_argument('--mosaic', type=float, default=0.6)
    parser.add_argument('--close_mosaic', type=int, default=20)
    parser.add_argument('--degrees', type=float, default=10.0)
    parser.add_argument('--translate', type=float, default=0.10)
    parser.add_argument('--scale', type=float, default=0.40)
    parser.add_argument('--flipud', type=float, default=0.5)
    parser.add_argument('--fliplr', type=float, default=0.5)

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    task = opt.task

    data_path = resolve_path(opt.data)
    model_conf = resolve_path(opt.config)
    project_path = resolve_path(opt.project)

    model = YOLO(model_conf)

    if task == "train":
        pretrained_weights = False
        weights = opt.weights.strip() if opt.weights else ""
        if model_conf.lower().endswith((".yaml", ".yml")) and weights.lower() not in ("", "none", "null"):
            local_weights = resolve_path(weights)
            weights_path = local_weights if os.path.exists(local_weights) else weights
            model.load(weights_path)
            pretrained_weights = local_weights if os.path.exists(local_weights) else weights_path
            print(f"Loaded pretrained weights: {weights_path}")

        train_args = {
            "data": data_path,
            "epochs": opt.epochs,
            "imgsz": opt.imgsz,
            "workers": opt.workers,
            "batch": opt.batch_size,
            "optimizer": opt.optimizer,
            "device": opt.device,
            "amp": opt.amp,
            "cache": opt.cache,
            "project": project_path,
            "name": opt.name,
            "pretrained": pretrained_weights,

            "lr0": opt.lr0,
            "lrf": opt.lrf,
            "weight_decay": opt.weight_decay,
            "patience": opt.patience,
            "module_warmup_epochs": opt.module_warmup_epochs,
            "cos_lr": opt.cos_lr,

            # SAR 图像一般是灰度/伪灰度，不建议强颜色增强
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.10,

            # SAR_AIRcraft 推荐增强
            "mosaic": opt.mosaic,
            "close_mosaic": opt.close_mosaic,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "degrees": opt.degrees,
            "translate": opt.translate,
            "scale": opt.scale,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": opt.flipud,
            "fliplr": opt.fliplr,

            "exist_ok": True,
            "save_period": 10,
            "plots": True,
        }

        model.train(**train_args)

    elif task == "val":
        val_args = {
            "data": data_path,
            "imgsz": opt.imgsz,
            "batch": opt.batch_size,
            "device": opt.device,
            "workers": opt.workers,
            "project": project_path,
            "name": opt.name,
            "conf": opt.conf,
            "iou": opt.iou,
            "half": opt.half,
            "dnn": opt.dnn,
            "plots": True,
        }

        model.val(**val_args)

    elif task == "test":
        test_args = {
            "data": data_path,
            "imgsz": opt.imgsz,
            "batch": opt.batch_size,
            "device": opt.device,
            "workers": opt.workers,
            "project": project_path,
            "name": opt.name,
            "conf": opt.conf,
            "iou": opt.iou,
            "half": opt.half,
            "dnn": opt.dnn,
            "split": "test",
            "plots": True,
        }

        model.val(**test_args)

    else:
        raise ValueError(f"Unsupported task: {task}")
