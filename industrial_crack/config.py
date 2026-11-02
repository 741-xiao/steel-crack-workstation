"""Validated, portable inspection settings."""

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Self


@dataclass(frozen=True)
class InspectionConfig:
    """Parameters saved with every inspection run; paths resolve beside the config."""

    weights: str = "runs/detect/steel_crack_detect_v8n_0808_1850/weights/best.pt"
    output: str = "inspection_data"
    confidence: float = 0.25
    iou: float = 0.45
    image_size: int = 512
    device: str = "cpu"
    video_stride: int = 1
    max_detections: int = 300
    save_images: str = "all"
    min_free_mb: int = 1024
    # Pixel coordinates [left, top, right, bottom]; null means the whole frame.
    roi: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        for name in ("weights", "output", "device"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} 必须是非空字符串")
        for name in ("confidence", "iou"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 < value <= 1
            ):
                raise ValueError(f"{name} 必须大于 0 且不超过 1")
        for name in ("image_size", "video_stride", "max_detections", "min_free_mb"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} 必须是正整数")
        if self.image_size % 32 or self.image_size > 4096:
            raise ValueError("image_size 必须是 32 的倍数且不超过 4096")
        if self.save_images not in ("all", "defects", "none"):
            raise ValueError("save_images 必须是 all/defects/none")
        if self.roi is not None:
            if (
                not isinstance(self.roi, (tuple, list))
                or len(self.roi) != 4
                or any(type(v) is not int for v in self.roi)
            ):
                raise ValueError("ROI 必须是四个整数")
            x1, y1, x2, y2 = self.roi
            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                raise ValueError("ROI 必须为有效的非负矩形")
            object.__setattr__(self, "roi", tuple(self.roi))

    def to_dict(self) -> dict[str, object]:
        """Return settings suitable for JSON persistence."""
        return asdict(self)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Read settings and resolve relative paths against their config file."""
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict):
            raise TypeError("配置必须是 JSON 对象")
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"未知配置项：{', '.join(sorted(unknown))}")
        cfg = cls(**raw)
        resolved = cfg.to_dict()
        for key in ("weights", "output"):
            candidate = Path(str(resolved[key])).expanduser()
            resolved[key] = str((path.resolve().parent / candidate).resolve())
        return cls(**resolved)  # type: ignore[arg-type]
