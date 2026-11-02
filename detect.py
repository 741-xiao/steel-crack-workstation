"""用训练好的模型做推理：图片 / 视频 / 文件夹。

自动识别模型类型（检测模型画框，分割模型画掩码），无需手动指定。

用法：
  python detect.py --weights runs/detect/steel_crack_detect_v8n_xxxx/best.pt --source test.jpg
  python detect.py --weights runs/segment/.../best.pt --source 视频.mp4
  python detect.py --weights .../best.pt --source 图片文件夹/ --conf 0.3

结果输出：runs/predict/（含标注图 + txt 结果）
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault('PYTHONUTF8', '1')

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description='钢结构裂缝推理')
    ap.add_argument('--weights', required=True, help='best.pt / last.pt 路径')
    ap.add_argument('--source', required=True, help='图片 / 视频 / 文件夹路径')
    ap.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    ap.add_argument('--iou', type=float, default=0.45, help='NMS IoU 阈值')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--out', default='runs/predict', help='结果输出目录')
    args = ap.parse_args()

    model = YOLO(args.weights)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        save=True,        # 保存标注图
        save_txt=True,    # 保存坐标结果
        save_conf=True,   # 结果里带置信度
        project=str(Path(args.out).parent),
        name=Path(args.out).name,
    )
    print(f'完成，共处理 {len(results)} 个输入，结果目录：{args.out}/')


if __name__ == '__main__':
    main()
