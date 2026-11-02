"""把原始图像 + 标签按比例切分为 train / val，并整理成 YOLO 目录结构。

用法：
  python split_data.py --images 原始图片目录 --labels 原始标签目录
  python split_data.py --images ... --labels ... --out datasets/steel_crack_dataset
  python split_data.py --images ... --labels ... --val-ratio 0.2 --seed 42
"""
import argparse
import random
import shutil
from pathlib import Path

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def find_pairs(images_dir, labels_dir):
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    pairs = []
    for img in images_dir.iterdir():
        if img.suffix.lower() not in IMG_EXT:
            continue
        label = labels_dir / (img.stem + '.txt')
        if label.exists():
            pairs.append((img, label))
    return pairs


def detect_label_format(label_path):
    """判断标签是检测格式(5列)还是分割格式(>=6列多边形)。返回 'box' / 'seg' / 'unknown'。"""
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n = len(line.split())
            if n == 5:
                return 'box'
            if n >= 6:
                return 'seg'
            return 'unknown'
    return 'unknown'


def main():
    ap = argparse.ArgumentParser(description='切分钢结构裂缝数据集为 train/val')
    ap.add_argument('--images', required=True, help='原始图片目录（含所有 jpg/png 等）')
    ap.add_argument('--labels', required=True, help='原始标签目录（对应 .txt 文件）')
    ap.add_argument('--out', default='datasets/steel_crack_dataset', help='输出数据集根目录')
    ap.add_argument('--val-ratio', type=float, default=0.2, help='验证集比例（默认 0.2）')
    ap.add_argument('--seed', type=int, default=42, help='随机种子，保证每次切分一致')
    args = ap.parse_args()

    pairs = find_pairs(args.images, args.labels)
    if not pairs:
        print('[错误] 没找到图片+标签配对，请检查 --images / --labels 路径。')
        return

    fmt = detect_label_format(pairs[0][1])
    print(f'共找到 {len(pairs)} 对图片+标签，标签格式：{fmt}')
    if fmt == 'unknown':
        print('[警告] 标签格式无法识别，仍将按原样复制。')

    random.seed(args.seed)
    random.shuffle(pairs)
    n_val = max(1, round(len(pairs) * args.val_ratio))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    out = Path(args.out)
    for split, items in (('train', train_pairs), ('val', val_pairs)):
        img_out = out / 'images' / split
        lab_out = out / 'labels' / split
        img_out.mkdir(parents=True, exist_ok=True)
        lab_out.mkdir(parents=True, exist_ok=True)
        for img, label in items:
            shutil.copy2(img, img_out / img.name)
            shutil.copy2(label, lab_out / label.name)

    print(f'完成：train={len(train_pairs)} 张，val={len(val_pairs)} 张')
    print(f'输出目录：{out.resolve()}')
    print('框标签可直接 train.py 训练检测；分割标签可直接训练 seg。')


if __name__ == '__main__':
    main()
