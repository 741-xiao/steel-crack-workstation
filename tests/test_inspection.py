"""Behavioral checks for evidence persistence and batch failure semantics."""

import csv
import json
import sqlite3
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from industrial_crack.config import InspectionConfig
from industrial_crack.service import infer, inspect_batch
from industrial_crack.storage import InspectionStore


class FakeTensor:
    """Minimal CPU tensor boundary used by the YOLO result adapter."""

    def __init__(self, values: list[object]) -> None:
        self.values = values

    def cpu(self) -> "FakeTensor":
        return self

    def tolist(self) -> list[object]:
        return self.values


class FakeModel:
    """Deterministic fake model for storage and coordinate tests."""

    task = "detect"
    names = {0: "crack"}

    def __init__(self, detections: bool = True) -> None:
        self.detections = detections

    def predict(self, source: np.ndarray, **kwargs: object) -> list[SimpleNamespace]:
        boxes = SimpleNamespace(
            xyxy=FakeTensor([[1, 2, 10, 12]] if self.detections else []),
            conf=FakeTensor([0.75] if self.detections else []),
            cls=FakeTensor([0] if self.detections else []),
        )
        return [
            SimpleNamespace(
                boxes=boxes, masks=None, names=self.names, plot=lambda: source.copy()
            )
        ]


def setup(tmp_path: Path, *, save_images: str = "all") -> tuple[InspectionConfig, Path]:
    """Create a local model stand-in and PNG with Unicode filename."""
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"local model")
    image = tmp_path / "裂缝样例.png"
    ok, encoded = cv2.imencode(".png", np.zeros((32, 32, 3), dtype=np.uint8))
    assert ok
    image.write_bytes(bytes(encoded))
    config = InspectionConfig(
        weights=str(weights),
        output=str(tmp_path / "output"),
        save_images=save_images,
        min_free_mb=1,
    )
    return config, image


def test_traceable_result_and_export(tmp_path: Path) -> None:
    config, image = setup(tmp_path)
    summary = inspect_batch(
        config, image, "=unsafe-batch", model_factory=lambda _: FakeModel()
    )
    assert summary.status == "COMPLETED"
    assert (summary.processed, summary.defects, summary.errors) == (1, 1, 0)
    with InspectionStore(Path(config.output)) as store:
        run = store.runs()[0]
        result = store.history(summary.run_id)[0]
        assert run["model_sha256"] and run["status"] == "COMPLETED"
        assert result["source_sha256"] and result["status"] == "DEFECT"
        assert json.loads(str(result["detections_json"]))[0]["xyxy"] == [1, 2, 10, 12]
        for field in ("original_path", "annotated_path"):
            assert (Path(config.output) / str(result[field])).is_file()
        export = tmp_path / "export.csv"
        assert store.export_csv(export, summary.run_id) == 1
    with export.open(encoding="utf-8-sig", newline="") as handle:
        assert next(csv.DictReader(handle))["batch"] == "'=unsafe-batch"


def test_undecodable_image_is_recorded_as_error(tmp_path: Path) -> None:
    config, image = setup(tmp_path)
    image.write_bytes(b"corrupt")
    summary = inspect_batch(
        config, image, "bad-image", model_factory=lambda _: FakeModel()
    )
    assert summary.status == "COMPLETED_WITH_ERRORS"
    assert (summary.processed, summary.errors) == (0, 1)
    with InspectionStore(Path(config.output)) as store:
        assert store.history(summary.run_id)[0]["status"] == "ERROR"


def test_cancel_stops_after_persisted_frame(tmp_path: Path) -> None:
    config, image = setup(tmp_path, save_images="none")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / image.name).write_bytes(image.read_bytes())
    second = inputs / "second.png"
    second.write_bytes(image.read_bytes())
    stop = Event()
    summary = inspect_batch(
        config,
        inputs,
        "cancel",
        stop=stop,
        callback=lambda _: stop.set(),
        model_factory=lambda _: FakeModel(False),
    )
    assert summary.status == "CANCELLED"
    assert summary.processed == 1
    with InspectionStore(Path(config.output)) as store:
        assert len(store.history(summary.run_id)) == 1
        assert store.history(summary.run_id)[0]["original_path"] is None


def test_roi_coordinates_and_invalid_config(tmp_path: Path) -> None:
    config, _ = setup(tmp_path)
    config = InspectionConfig(**{**config.to_dict(), "roi": (5, 6, 20, 24)})
    detections, annotated = infer(
        FakeModel(), np.zeros((32, 32, 3), dtype=np.uint8), config
    )
    assert detections[0]["xyxy"] == [6, 8, 15, 18]
    assert annotated.shape == (32, 32, 3)
    with pytest.raises(ValueError, match="image_size"):
        InspectionConfig(image_size=513)
    with pytest.raises(ValueError, match="ROI"):
        InspectionConfig(roi=(2, 3, 1, 5))


def test_model_load_failure_marks_run_failed(tmp_path: Path) -> None:
    config, image = setup(tmp_path)
    with pytest.raises(RuntimeError, match="cannot load"):
        inspect_batch(
            config,
            image,
            "failed",
            model_factory=lambda _: (_ for _ in ()).throw(RuntimeError("cannot load")),
        )
    with InspectionStore(Path(config.output)) as store:
        assert store.runs()[0]["status"] == "FAILED"


def test_video_stride_and_source_hash(tmp_path: Path) -> None:
    config, _ = setup(tmp_path, save_images="none")
    config = InspectionConfig(**{**config.to_dict(), "video_stride": 2})
    video = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (32, 32))
    assert writer.isOpened()
    for value in (20, 40, 60):
        writer.write(np.full((32, 32, 3), value, dtype=np.uint8))
    writer.release()
    summary = inspect_batch(
        config, video, "video", model_factory=lambda _: FakeModel(False)
    )
    assert summary.status == "COMPLETED" and summary.processed == 2
    with InspectionStore(Path(config.output)) as store:
        records = store.history(summary.run_id)
    assert {record["frame_index"] for record in records} == {0, 2}
    assert records[0]["source_sha256"] == records[1]["source_sha256"]


def test_storage_failure_marks_run_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, image = setup(tmp_path)

    def reject(*args: object, **kwargs: object) -> None:
        raise sqlite3.OperationalError("disk unavailable")

    monkeypatch.setattr(InspectionStore, "record", reject)
    with pytest.raises(RuntimeError, match="disk unavailable"):
        inspect_batch(
            config, image, "storage-error", model_factory=lambda _: FakeModel()
        )
    with InspectionStore(Path(config.output)) as store:
        assert store.runs()[0]["status"] == "FAILED"


@pytest.mark.parametrize(
    "setting,value",
    [
        ("confidence", 0),
        ("iou", float("nan")),
        ("video_stride", 0),
        ("max_detections", -1),
        ("save_images", "unexpected"),
        ("roi", [1, 2, 0, 4]),
    ],
)
def test_invalid_config_values(setting: str, value: object) -> None:
    with pytest.raises(ValueError):
        InspectionConfig(**{setting: value})


def test_load_config_resolves_paths_and_rejects_unknown(tmp_path: Path) -> None:
    path = tmp_path / "cfg.json"
    path.write_text(
        json.dumps({"weights": "model.pt", "output": "out"}), encoding="utf-8"
    )
    config = InspectionConfig.load(path)
    assert config.weights == str(tmp_path / "model.pt")
    assert config.output == str(tmp_path / "out")
    path.write_text('{"unexpected": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="未知"):
        InspectionConfig.load(path)
