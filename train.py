"""训练钢结构裂缝检测 / 分割模型。

用法：
  python train.py                          # 默认：yolov8n 检测，120 epoch，设备自动选
  python train.py --task seg               # 训练分割模型（需要分割标签）
  python train.py --model s --epochs 60    # 换大一点的模型 / 少训几轮
  python train.py --device cpu             # 强制 CPU（无独显时默认就是 CPU）

输出：runs/{detect|segment}/steel_crack_xxx_日期时间/
      best.pt / last.pt / results.png / PR_curve.png 等
"""
import argparse
import os
from datetime import datetime

os.environ.setdefault('PYTHONUTF8', '1')

from ultralytics import YOLO


def pick_device():
    import torch
    return 0 if torch.cuda.is_available() else 'cpu'


def main():
    ap = argparse.ArgumentParser(description='钢结构裂缝检测/分割训练')
    ap.add_argument('--task', choices=['detect', 'seg'], default='detect',
                    help='detect=画框检测，seg=分割')
    ap.add_argument('--model', choices=['n', 's', 'm', 'l', 'x'], default='n',
                    help='基础模型体积，n 最轻量（CPU 建议 n/s）')
    ap.add_argument('--data', default='dataset.yaml', help='数据集配置文件')
    ap.add_argument('--epochs', type=int, default=120, help='训练轮数')
    ap.add_argument('--batch', type=int, default=8, help='批大小，CPU 建议 8 或更小')
    ap.add_argument('--imgsz', type=int, default=640, help='输入分辨率')
    ap.add_argument('--device', default=None, help='cpu 或 cuda，默认自动')
    ap.add_argument('--workers', type=int, default=0, help='数据加载线程，Windows 保持 0')
    ap.add_argument('--lr0', type=float, default=0.001, help='初始学习率')
    ap.add_argument('--mosaic', type=float, default=0.5, help='马赛克增强比例')
    ap.add_argument('--mixup', type=float, default=0.2, help='MixUp 增强比例')
    args = ap.parse_args()

    device = args.device or pick_device()
    weights = f'yolov8{args.model}-seg.pt' if args.task == 'seg' else f'yolov8{args.model}.pt'
    print(f'任务：{args.task} | 模型：{weights} | 设备：{device}')

    model = YOLO(weights)

    name = f'steel_crack_{args.task}_v8{args.model}_{datetime.now():%m%d_%H%M}'
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        optimizer='AdamW',
        lr0=args.lr0,
        weight_decay=0.0005,
        mosaic=args.mosaic,
        mixup=args.mixup,
        save_period=10,
        val=True,
        plots=True,
        name=name,
    )

    print(f'训练完成，模型权重保存在 runs/{args.task}/steel_crack_{args.task}_v8{args.model}_日期时间/ 下的 best.pt / last.pt。')


if __name__ == '__main__':
    main()
