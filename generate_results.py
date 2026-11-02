"""训练完成后自动生成结果展示。

用法：
  python generate_results.py --run runs/detect/steel_crack_detect_v8n_0808_0839
  python generate_results.py --run runs/segment/xxx --data dataset_seg.yaml

生成到 结果展示/ 文件夹：
  - 训练曲线（loss、mAP 等）
  - PR 曲线、F1 曲线、混淆矩阵
  - 用 best.pt 检测的样例图（画框/掩码标出裂缝）
"""
import argparse
import os
import random
import shutil
from pathlib import Path

os.environ.setdefault('PYTHONUTF8', '1')

import cv2
import yaml
from ultralytics import YOLO

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def main():
    ap = argparse.ArgumentParser(description='生成训练结果展示')
    ap.add_argument('--run', required=True, help='训练输出目录（含 weights/best.pt）')
    ap.add_argument('--data', default='fast_detect.yaml', help='评估用数据集配置')
    ap.add_argument('--samples', type=int, default=12, help='展示的检测样例图数量')
    ap.add_argument('--out', default='结果展示', help='输出文件夹')
    args = ap.parse_args()

    run = Path(args.run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    best = run / 'weights' / 'best.pt'
    if not best.exists():
        print(f'[错误] 找不到 {best}，请确认训练已完成。')
        return
    model = YOLO(str(best))
    task = getattr(model, 'task', 'detect')
    print(f'加载模型：{best.name}（{task}）')

    # ---- 1. 复制训练生成的曲线图 ----
    curve_files = [
        'results.png', 'PR_curve.png', 'F1_curve.png',
        'confusion_matrix.png', 'confusion_matrix_normalized.png',
        'val_batch0_labels.jpg', 'val_batch0_pred.jpg',
    ]
    for name in curve_files:
        src = run / name
        if src.exists():
            shutil.copy2(src, out / f'{task}_{name}')
            print(f'  复制曲线图：{name}')

    # ---- 2. 用 best.pt 检测验证集样例 ----
    with open(args.data, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    base = Path(args.data).resolve().parent / cfg['path']
    val_dir = base / cfg['val']

    all_imgs = [p for p in val_dir.iterdir() if p.suffix.lower() in IMG_EXT]
    if not all_imgs:
        print(f'[错误] 验证集目录为空：{val_dir}')
        return
    if len(all_imgs) > args.samples:
        random.seed(42)
        sample_imgs = random.sample(all_imgs, args.samples)
    else:
        sample_imgs = all_imgs

    results = model(sample_imgs, conf=0.25, imgsz=512, device='cpu')

    sample_out = out / f'{task}_检测样例'
    sample_out.mkdir(parents=True, exist_ok=True)
    total_det = 0
    for res in results:
        boxes = res.boxes
        if boxes is not None:
            total_det += len(boxes)
        annotated = res.plot()
        # 用字节流写图，避免中文路径被 cv2.imwrite 拒绝
        cv2.imencode('.jpg', annotated)[1].tofile(sample_out / f'{Path(res.path).stem}.jpg')

    print(f'  检测样例：{len(results)} 张，输出到 {sample_out}/')
    print(f'  共检测到裂缝 {total_det} 个（{len(sample_imgs)} 张样例）')

    print(f'结果展示生成完成，输出到：{out.resolve()}')


if __name__ == '__main__':
    main()
