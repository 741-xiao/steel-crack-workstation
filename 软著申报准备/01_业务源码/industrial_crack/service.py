"""Bounded-memory offline inspection with cooperative cancellation."""

import hashlib
import importlib.metadata
import logging
import platform
import shutil
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from time import perf_counter
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from . import __version__
from .config import InspectionConfig
from .storage import InspectionStore

Image = NDArray[np.uint8]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
LOGGER = logging.getLogger(__name__)


@dataclass
class InspectionEvent:
    """An already committed result and optional preview for the caller."""

    record: dict[str, object]
    preview: Image | None


@dataclass(frozen=True)
class RunSummary:
    """Run-level outcome; no detections never implies product acceptance."""

    run_id: str
    status: str
    processed: int
    defects: int
    errors: int


def sha256_file(path: Path) -> str:
    """Hash local weights without loading them entirely into memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def encode_png(image: Image) -> bytes:
    """Losslessly encode a decoded original or annotated image."""
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise OSError("图像编码失败")
    return bytes(encoded)


def sources(path: Path) -> Iterator[Path]:
    """Enumerate supported inputs deterministically, excluding output by caller."""
    path = path.resolve()
    if path.is_dir():
        found = False
        for item in sorted(path.rglob("*")):
            if (
                item.is_file()
                and item.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
            ):
                found = True
                yield item
        if not found:
            raise ValueError("目录中没有支持的图片或视频")
    elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
        yield path
    else:
        raise ValueError(f"输入不存在或格式不支持：{path}")


def read_frames(
    path: Path, stride: int, stop: Event
) -> Generator[tuple[int | None, Image, str]]:
    """Read Unicode image paths or video frames; retain at most one decoded frame."""
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        raw = path.read_bytes()
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("图片无法解码")
        yield None, cast(Image, image), hashlib.sha256(raw).hexdigest()
        return
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("视频无法打开")
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        index = 0
        while not stop.is_set():
            success, image = capture.read()
            if not success:
                if index == 0 or (expected > 0 and index < expected):
                    raise ValueError(
                        f"视频提前结束或解码失败：已读 {index} / {expected} 帧"
                    )
                break
            if index % stride == 0:
                yield index, cast(Image, image), hashlib.sha256(
                    image.tobytes()
                ).hexdigest()
            index += 1
    finally:
        capture.release()


def infer(
    model: Any, image: Image, config: InspectionConfig
) -> tuple[list[dict[str, object]], Image]:
    """Predict a frame or ROI; all stored coordinates refer to the original frame."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = config.roi or (0, 0, width, height)
    if x2 > width or y2 > height:
        raise ValueError(f"ROI 超出图片范围 {width}×{height}")
    result = model.predict(
        source=image[y1:y2, x1:x2],
        conf=config.confidence,
        iou=config.iou,
        imgsz=config.image_size,
        device=config.device,
        max_det=config.max_detections,
        verbose=False,
        save=False,
    )[0]
    detections: list[dict[str, object]] = []
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()
        masks = result.masks.xy if result.masks is not None else None
        for index, (box, confidence, class_id) in enumerate(
            zip(boxes, confidences, classes, strict=True)
        ):
            detection: dict[str, object] = {
                "class_id": int(class_id),
                "class_name": str(result.names[int(class_id)]),
                "confidence": float(confidence),
                "xyxy": [
                    float(box[0] + x1),
                    float(box[1] + y1),
                    float(box[2] + x1),
                    float(box[3] + y1),
                ],
            }
            if masks is not None:
                detection["polygon"] = (masks[index] + np.array([x1, y1])).tolist()
            detections.append(detection)
    annotated = image.copy()
    annotated[y1:y2, x1:x2] = result.plot()
    if config.roi:
        cv2.rectangle(annotated, (x1, y1), (x2 - 1, y2 - 1), (0, 180, 255), 1)
    return detections, annotated


def inspect_batch(
    config: InspectionConfig,
    source: Path,
    batch: str,
    stop: Event | None = None,
    callback: Callable[[InspectionEvent], None] | None = None,
    model_factory: Callable[[str], Any] | None = None,
) -> RunSummary:
    """Inspect local inputs and durably record every outcome before notifying UI.

    Input/model errors are recorded; storage failures abort the run. Cancellation
    takes effect between frames, never through an unsafe thread termination.
    """
    stop = stop if stop is not None else Event()
    weights, output, source = (
        Path(config.weights).resolve(),
        Path(config.output).resolve(),
        source.resolve(),
    )
    if not weights.is_file():
        raise FileNotFoundError(f"找不到本地模型：{weights}")
    if (
        source == output
        or output in source.parents
        or (source.is_dir() and source in output.parents)
    ):
        raise ValueError("输入目录与结果目录不能相互包含，以防再次检测输出图片")
    if not source.exists():
        raise FileNotFoundError(f"输入不存在：{source}")
    if not batch.strip():
        raise ValueError("批次名称不能为空")
    model_hash = sha256_file(weights)
    versions = {
        "app": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in ("ultralytics", "torch", "opencv-python", "numpy"):
        versions[package] = importlib.metadata.version(package)
    processed = defects = errors = 0
    with InspectionStore(output) as store:
        run_id = store.begin(batch, model_hash, config.to_dict(), versions)
        try:

            def check_space() -> None:
                if shutil.disk_usage(output).free < config.min_free_mb * 1024 * 1024:
                    raise OSError("结果磁盘剩余空间低于配置下限，检测已停止")

            check_space()
            if model_factory is None:
                from ultralytics import YOLO  # type: ignore[attr-defined]

                model_factory = YOLO
            model = model_factory(str(weights))
            if sha256_file(weights) != model_hash:
                raise RuntimeError("模型在加载过程中发生改变，请停止修改模型后重试")
            if getattr(model, "task", "detect") not in ("detect", "segment"):
                raise ValueError("仅支持 YOLO 检测或分割模型")
            # This application is dedicated to single-class crack models.
            if len(model.names) != 1 or str(model.names[0]).lower() != "crack":
                raise ValueError("请选择仅包含 crack 类别的钢结构裂缝模型")
            for item in sources(source):
                if stop.is_set():
                    break
                frames = read_frames(item, config.video_stride, stop)
                video_hash = (
                    sha256_file(item)
                    if item.suffix.lower() in VIDEO_EXTENSIONS
                    else None
                )
                frame_index: int | None = None
                try:
                    while not stop.is_set():
                        try:
                            frame_index, original, frame_hash = next(frames)
                        except StopIteration:
                            break
                        source_hash = video_hash or frame_hash
                        started = perf_counter()
                        detections, annotated = infer(model, original, config)
                        elapsed = (perf_counter() - started) * 1000
                        status = "DEFECT" if detections else "NO_DETECTION"
                        check_space()
                        save = config.save_images == "all" or (
                            config.save_images == "defects" and bool(detections)
                        )
                        # Store/encoding failures must escape the input-error handler.
                        try:
                            row = store.record(
                                run_id,
                                str(item),
                                frame_index,
                                source_hash,
                                status,
                                detections,
                                elapsed,
                                encode_png(original) if save else None,
                                encode_png(annotated) if save else None,
                            )
                        except Exception as error:
                            raise StorageError(str(error)) from error
                        processed += 1
                        defects += bool(detections)
                        if callback:
                            callback(InspectionEvent(row, annotated))
                except (StorageError, OSError):
                    raise
                except Exception as error:
                    LOGGER.exception("输入处理失败：%s", item)
                    errors += 1
                    row = store.record(
                        run_id,
                        str(item),
                        frame_index,
                        "",
                        "ERROR",
                        [],
                        0,
                        error=str(error),
                    )
                    if callback:
                        callback(InspectionEvent(row, None))
                finally:
                    frames.close()
            status = (
                "CANCELLED"
                if stop.is_set()
                else "COMPLETED_WITH_ERRORS" if errors else "COMPLETED"
            )
            store.finish(run_id, status)
            return RunSummary(run_id, status, processed, defects, errors)
        except Exception as error:
            store.finish(run_id, "FAILED", str(error))
            raise


class StorageError(RuntimeError):
    """A persistence error that must stop inspection rather than discard evidence."""
