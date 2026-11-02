"""Repeatable YOLO validation for comparing input sizes on the same images.

Examples:
  python accuracy_experiment.py --limit 80 --imgsz 512
  python accuracy_experiment.py --limit 80 --imgsz 640
  python accuracy_experiment.py --limit 0 --imgsz 640

The script never modifies the deployed model or validation labels.
"""

import argparse
import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "datasets" / "steel_crack_dataset"
DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "steel_crack_detect_v8n_0808_1850" / "weights" / "best.pt"
OUTPUT = ROOT / "verification_output" / "accuracy_experiments"


def sha256_file(path: Path) -> str:
    """Identify the exact evaluated model."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    """Evaluate a fixed subset or all 530 validation images."""
    parser = argparse.ArgumentParser(description="裂缝模型精度对比实验")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--limit", type=int, default=80, help="0=全部验证集；其他值=固定抽样张数")
    parser.add_argument("--imgsz", type=int, choices=(512, 640, 768), default=512)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    if args.limit < 0 or args.batch < 1:
        parser.error("limit 不能为负数，batch 必须大于 0")
    weights = args.weights.resolve()
    if not weights.is_file():
        parser.error(f"模型文件不存在：{weights}")
    images = sorted((DATASET / "images" / "val").glob("*.png"))
    if args.limit:
        images = random.Random(42).sample(images, min(args.limit, len(images)))
    if not images:
        parser.error("未找到验证图片")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    subset = OUTPUT / f"val_{len(images)}_seed42.txt"
    subset.write_text("\n".join(path.resolve().as_posix() for path in images) + "\n", encoding="utf-8")
    data = OUTPUT / f"dataset_{len(images)}_seed42.yaml"
    data.write_text(
        yaml.safe_dump({"path": DATASET.resolve().as_posix(), "train": "images/train",
                        "val": subset.resolve().as_posix(), "names": {0: "crack"}}, allow_unicode=True),
        encoding="utf-8",
    )
    started = perf_counter()
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data), imgsz=args.imgsz, batch=args.batch, device="cpu", workers=0,
        plots=False, save_json=False, project=str(OUTPUT),
        name=f"val_{len(images)}_img{args.imgsz}", exist_ok=True, verbose=False,
    )
    report = {
        "utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "weights": str(weights), "weights_sha256": sha256_file(weights),
        "validation_images": len(images), "selection": "all" if not args.limit else "seed42_random",
        "selection_file_sha256": sha256_file(subset), "imgsz": args.imgsz,
        "batch": args.batch, "device": "cpu",
        "elapsed_seconds": round(perf_counter() - started, 1),
        "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
    }
    destination = OUTPUT / f"report_{len(images)}_img{args.imgsz}.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"报告：{destination}")


if __name__ == "__main__":
    main()
