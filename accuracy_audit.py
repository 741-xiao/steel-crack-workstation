"""Audit YOLO crack labels and training history before changing a model.

Usage: python accuracy_audit.py
The script is read-only and uses the existing train/validation split.
"""

import csv
import hashlib
import json
from pathlib import Path
from statistics import median

from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "steel_crack_dataset"
RESULTS = ROOT / "runs" / "detect" / "steel_crack_detect_v8n_0808_1850" / "results.csv"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def audit_split(split: str) -> tuple[dict[str, object], set[str]]:
    """Check label syntax, box size, and file pairings in one split."""
    images = sorted(path for path in (DATA / "images" / split).iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    labels = DATA / "labels" / split
    sizes: list[tuple[float, float]] = []
    hashes: set[str] = set()
    empty = 0
    invalid: list[str] = []
    duplicates = 0
    boxes_per_image: list[int] = []
    for image in images:
        label = labels / f"{image.stem}.txt"
        if not label.is_file():
            invalid.append(f"缺少标签：{image.name}")
            continue
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if digest in hashes:
            duplicates += 1
        hashes.add(digest)
        with Image.open(image) as opened:
            width, height = opened.size
        lines = label.read_text(encoding="utf-8-sig").splitlines()
        if not lines:
            empty += 1
        boxes_per_image.append(len(lines))
        for line_number, line in enumerate(lines, 1):
            try:
                values = list(map(float, line.split()))
                if len(values) != 5 or values[0] != 0:
                    raise ValueError("预期单类 YOLO 框标签")
                cx, cy, bw, bh = values[1:]
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                    raise ValueError("坐标超出范围")
                if cx - bw / 2 < -0.00001 or cx + bw / 2 > 1.00001:
                    raise ValueError("横向边界超出图片")
                if cy - bh / 2 < -0.00001 or cy + bh / 2 > 1.00001:
                    raise ValueError("纵向边界超出图片")
                sizes.append((bw * width, bh * height))
            except ValueError as error:
                invalid.append(f"{label.name}:{line_number} {error}")
    widths = [width for width, _ in sizes]
    heights = [height for _, height in sizes]
    thin = sum(min(width, height) < 8 for width, height in sizes)
    tiny = sum(width * height < 64 for width, height in sizes)
    return {
        "images": len(images),
        "label_files": len(list(labels.glob("*.txt"))),
        "empty_label_images": empty,
        "boxes": len(sizes),
        "boxes_per_image_median": median(boxes_per_image) if boxes_per_image else 0,
        "box_width_px_median": round(median(widths), 1) if widths else 0,
        "box_height_px_median": round(median(heights), 1) if heights else 0,
        "thin_side_under_8px": thin,
        "thin_side_under_8px_percent": round(100 * thin / len(sizes), 1) if sizes else 0,
        "area_under_64px2_percent": round(100 * tiny / len(sizes), 1) if sizes else 0,
        "duplicate_images_within_split": duplicates,
        "invalid_labels": invalid[:20],
        "invalid_label_count": len(invalid),
    }, hashes


def main() -> None:
    """Print a machine-readable audit summary."""
    train, train_hashes = audit_split("train")
    val, val_hashes = audit_split("val")
    with RESULTS.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    best = max(rows, key=lambda row: float(row["metrics/mAP50(B)"]))
    last = rows[-1]
    report = {
        "train": train,
        "val": val,
        "identical_images_across_splits": len(train_hashes & val_hashes),
        "best_epoch_by_map50": int(best["epoch"]),
        "best_epoch_map50": float(best["metrics/mAP50(B)"]),
        "last_epoch": int(last["epoch"]),
        "last_epoch_map50": float(last["metrics/mAP50(B)"]),
        "last_epoch_elapsed_hours": round(float(last["time"]) / 3600, 1),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
