"""从掩码(mask)生成 YOLO 标签，并整理成标准数据集。

原始数据结构：
  image_Steelcrack钢结构YHT4355/
    Train/ {images/, masks/, edges/}
    Test/  {images/, masks/, edges/, txt/}

本脚本为每一张图生成两种标签：
  - 检测框标签（labels/*.txt）：class cx cy w h，从裂缝连通域的边界框得到
  - 分割标签（seg_labels/*.txt）：class x1 y1 x2 y2 ...，从裂缝轮廓多边形得到

输出两个数据集（图片只复制一份，分割数据集用目录联接复用图片）：
  datasets/steel_crack_dataset/   # 检测：images + labels
  datasets/steel_crack_seg/       # 分割：images(联接) + labels

切分：默认按原始 Train/Test 划分（Train 训练，Test 验证）。
也可以 --random-split 在全部数据上随机 80/20 切分。

用法：
  python convert_masks_to_labels.py
  python convert_masks_to_labels.py --random-split --seed 42
"""
import argparse
import random
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
RAW_ROOT = Path(__file__).parent / 'image_Steelcrack钢结构YHT4355'
DET_OUT = Path(__file__).parent / 'datasets' / 'steel_crack_dataset'
SEG_OUT = Path(__file__).parent / 'datasets' / 'steel_crack_seg'


def imread_unicode(path):
    """Windows 下 cv2.imread 无法读中文路径，改用字节流解码。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def load_mask(mask_path):
    """读取掩码，返回二值 uint8 图（裂缝=255）。"""
    m = imread_unicode(mask_path)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[:, :, 0]
    _, binary = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return (binary > 0).astype(np.uint8) * 255


def mask_to_labels(mask_path, min_area_ratio=0.0001):
    """从掩码生成 (框行列表, 分割行列表)。"""
    binary = load_mask(mask_path)
    if binary is None:
        return None, None
    h, w = binary.shape[:2]
    min_pixels = max(1, int(w * h * min_area_ratio))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    box_lines = []
    seg_lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_pixels:
            continue
        # ---- 检测框 ----
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 1 or bh < 1:
            continue
        cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
        bw_n, bh_n = bw / w, bh / h
        if cx < 0 or cy < 0 or bw_n <= 0 or bh_n <= 0:
            continue
        box_lines.append(f'0 {cx:.6f} {cy:.6f} {bw_n:.6f} {bh_n:.6f}')

        # ---- 分割多边形 ----
        peri = cv2.arcLength(cnt, True)
        eps = max(0.0001, 0.002 * peri)
        approx = cv2.approxPolyDP(cnt, eps, True)
        pts = approx.reshape(-1, 2).astype(np.float32)
        pts[:, 0] = np.clip(pts[:, 0] / w, 0.0, 1.0)
        pts[:, 1] = np.clip(pts[:, 1] / h, 0.0, 1.0)
        if len(pts) < 3:
            continue
        seg_lines.append('0 ' + ' '.join(f'{v:.6f}' for v in pts.flatten().tolist()))
    return box_lines, seg_lines


def write_labels(lines, txt_path):
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + ('\n' if lines else ''))


def main():
    ap = argparse.ArgumentParser(description='掩码转 YOLO 标签并整理数据集')
    ap.add_argument('--min-area', type=float, default=0.0001,
                    help='最小连通域面积（占图像比例），用于滤除噪点')
    ap.add_argument('--random-split', action='store_true',
                    help='在全部图片上随机 80/20 切分，否则用原始 Train/Test 划分')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    if not RAW_ROOT.is_dir():
        print(f'[错误] 找不到原始数据目录：{RAW_ROOT}')
        return

    # ---- 1. 收集所有 图片+掩码 配对 ----
    items = []
    for split in ('Train', 'Test'):
        img_dir = RAW_ROOT / split / 'images'
        mask_dir = RAW_ROOT / split / 'masks'
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            mask = mask_dir / (img.stem + '.png')
            if mask.exists():
                items.append((img, mask, split))
    if not items:
        print('[错误] 没找到图片+掩码配对。')
        return
    print(f'共找到 {len(items)} 张带掩码的图片（Train/Test 自动合并）。')

    # ---- 2. 生成标签 ----
    records = []  # (img, split, box_lines, seg_lines)
    empty = 0
    for img, mask, split in items:
        boxes, segs = mask_to_labels(mask, args.min_area)
        if boxes is None:
            print(f'[警告] 掩码读取失败，跳过：{mask.name}')
            continue
        if not boxes:
            empty += 1
        records.append((img, split, boxes, segs))
    print(f'标签生成完成；无裂缝的图片 {empty} 张（标签为空文件）。')

    # ---- 3. 切分 ----
    if args.random_split:
        random.seed(args.seed)
        random.shuffle(records)
        n_val = round(len(records) * 0.2)
        val_records = records[:n_val]
        train_records = records[n_val:]
    else:
        train_records = [r for r in records if r[1] == 'Train']
        val_records = [r for r in records if r[1] == 'Test']
    print(f'训练集 {len(train_records)} 张，验证集 {len(val_records)} 张。')

    # ---- 4. 写检测数据集（复制图片 + 框标签） ----
    def copy_images(out_root):
        for split_recs, split in ((train_records, 'train'), (val_records, 'val')):
            img_out = out_root / 'images' / split
            img_out.mkdir(parents=True, exist_ok=True)
            existing = set(p.name for p in img_out.iterdir())
            copied = 0
            for img, _, _, _ in split_recs:
                if img.name not in existing:
                    shutil.copy2(img, img_out / img.name)
                    copied += 1
            print(f'  {split}: 复制 {copied} 张（已有 {len(existing)} 张跳过）')

    def write_labels_for(out_root, label_writer):
        for split_recs, split in ((train_records, 'train'), (val_records, 'val')):
            lab_out = out_root / 'labels' / split
            lab_out.mkdir(parents=True, exist_ok=True)
            for img, _, box_lines, seg_lines in split_recs:
                write_labels(label_writer(box_lines, seg_lines),
                             lab_out / (img.stem + '.txt'))

    copy_images(DET_OUT)
    write_labels_for(DET_OUT, lambda b, s: b)
    print(f'检测数据集完成：{DET_OUT.resolve()}')

    # ---- 5. 写分割数据集（联接复用图片 + 多边形标签） ----
    det_images = DET_OUT / 'images'
    seg_images = SEG_OUT / 'images'
    if not seg_images.exists():
        # 优先用目录联接(Windows)复用图片，避免重复占磁盘
        r = subprocess.run(['cmd', '/c', 'mklink', '/J', str(seg_images),
                            str(det_images)], capture_output=True, text=True,
                           errors='replace')
        if r.returncode != 0:
            print(f'[提示] 目录联接失败，改为复制图片：{r.stderr.strip()}')
            shutil.copytree(det_images / 'train', seg_images / 'train')
            shutil.copytree(det_images / 'val', seg_images / 'val')
    write_labels_for(SEG_OUT, lambda b, s: s)
    print(f'分割数据集完成：{SEG_OUT.resolve()}')

    print('全部完成。')
    print('  检测训练：python train.py --task detect')
    print('  分割训练：python train.py --task seg')


if __name__ == '__main__':
    main()
