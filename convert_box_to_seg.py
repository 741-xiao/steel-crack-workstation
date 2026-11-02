"""把检测格式(矩形框)标签转换为分割格式(4 点多边形)标签，用于训练 seg 模型。

说明：
  - 检测标签每行：class cx cy w h
  - 分割标签每行：class x1 y1 x2 y2 ... xn yn
  本脚本把每个框的 4 个角点写成分割标签，等价于一个矩形掩码。
  只有纯框格式的 .txt 会被转换，已经是分割格式的自动跳过。

用法：
  python convert_box_to_seg.py --labels datasets/steel_crack_dataset/labels
  python convert_box_to_seg.py --labels datasets/steel_crack_dataset/labels --backup
"""
import argparse
import shutil
from pathlib import Path


def box_to_seg(nums):
    cx, cy, w, h = nums
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def main():
    ap = argparse.ArgumentParser(description='框标签 -> 分割标签')
    ap.add_argument('--labels', required=True, help='标签目录（会就地改写 .txt）')
    ap.add_argument('--backup', action='store_true', help='转换前把原标签备份到 _box_backup/')
    args = ap.parse_args()

    labels = Path(args.labels)
    if not labels.is_dir():
        print(f'[错误] 标签目录不存在：{labels}')
        return

    backup_dir = labels / '_box_backup'
    if args.backup:
        backup_dir.mkdir(exist_ok=True)

    converted = skipped = 0
    for txt in labels.glob('*.txt'):
        with open(txt, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_lines = []
        all_box = True
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                new_lines.append(line)
                all_box = False
                continue
            cls = parts[0]
            seg = box_to_seg([float(v) for v in parts[1:]])
            new_lines.append(cls + ' ' + ' '.join(f'{v:.6f}' for v in seg) + '\n')

        if all_box:
            if args.backup:
                shutil.copy2(txt, backup_dir / txt.name)
            with open(txt, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            converted += 1
        else:
            skipped += 1

    print(f'转换完成：{converted} 个标签转为分割格式，跳过 {skipped} 个非框格式。')
    if args.backup:
        print(f'原标签已备份到：{backup_dir}')


if __name__ == '__main__':
    main()
