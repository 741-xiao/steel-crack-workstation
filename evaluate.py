"""在验证集上评估模型，输出 mAP / Precision / Recall 等指标。

用法：
  python evaluate.py --weights runs/detect/steel_crack_detect_v8n_xxxx/best.pt
  python evaluate.py --weights runs/segment/.../best.pt --data dataset.yaml

结果：终端打印指标，并保存 PR 曲线等图到 runs/detect/val*/ 或 runs/segment/val*/
"""
import argparse
import os

os.environ.setdefault('PYTHONUTF8', '1')

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description='验证集评估')
    ap.add_argument('--weights', required=True, help='best.pt 路径')
    ap.add_argument('--data', default='dataset.yaml')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.001, help='评估时阈值放低以算全指标')
    ap.add_argument('--iou', type=float, default=0.6)
    args = ap.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        plots=True,
        save_json=True,
    )
    print('评估完成。')
    print('指标：')
    for k, v in metrics.results_dict.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
