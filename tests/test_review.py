"""Human decisions must remain separate from predictions and reusable evidence."""

import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from industrial_crack.review import export_review_bundle, review_category
from industrial_crack.storage import InspectionStore


def seed(
    store: InspectionStore, status: str = "DEFECT", original: bytes | None = b"image"
) -> dict:
    run = store.begin("工件一", "model-hash", {"roi": None}, {"app": "2.0.0"})
    return store.record(run, "原始.png", None, "source-hash", status, [], 1, original)


def test_old_database_adds_reviews_without_changing_inspections(tmp_path: Path) -> None:
    with InspectionStore(tmp_path) as store:
        item = seed(store)
        # Emulate pre-review schema with an existing inspection and original.
        store.connection.execute("DROP TABLE reviews")
    with InspectionStore(tmp_path) as store:
        row = store.history()[0]
        assert row["id"] == item["id"] and row["review_version"] == 0
        assert row["review_decision"] == "PENDING"
        first = store.add_review(item["id"], "CLEAR", "张三", "反光而非裂缝", 0)
        store.add_review(item["id"], "UNCERTAIN", "李四", "需要更清晰照片", first)
        assert [r["decision"] for r in store.review_history(item["id"])] == [
            "CLEAR",
            "UNCERTAIN",
        ]
        assert store.history()[0]["status"] == "DEFECT"
        assert store.history(review_decision="CLEAR") == []
        assert len(store.history(review_decision="UNCERTAIN")) == 1
        assert (tmp_path / item["original_path"]).read_bytes() == b"image"


def test_stale_review_rejected_across_connections(tmp_path: Path) -> None:
    with InspectionStore(tmp_path) as first, InspectionStore(tmp_path) as second:
        item = seed(first)
        first.add_review(item["id"], "CONFIRMED", "张三", "已检查全图", 0)
        with pytest.raises(RuntimeError, match="新的复核"):
            second.add_review(item["id"], "CLEAR", "李四", "另一判断", 0)
        assert len(second.review_history(item["id"])) == 1


@pytest.mark.parametrize(
    "decision,reviewer,note",
    [
        ("PENDING", "甲", "说明"),
        ("CLEAR", " ", "说明"),
        ("CLEAR", "甲", " "),
        ("UNKNOWN", "甲", "说明"),
        ("CLEAR", "甲", "字" * 2001),
    ],
)
def test_invalid_review_is_not_persisted(
    tmp_path: Path, decision: str, reviewer: str, note: str
) -> None:
    with InspectionStore(tmp_path) as store:
        item = seed(store)
        with pytest.raises(ValueError):
            store.add_review(item["id"], decision, reviewer, note, 0)
        assert store.review_history(item["id"]) == []


def test_error_and_missing_record_cannot_be_reviewed(tmp_path: Path) -> None:
    with InspectionStore(tmp_path) as store:
        item = seed(store, "ERROR")
        for record_id in (item["id"], "does-not-exist"):
            with pytest.raises(ValueError, match="重新检测"):
                store.add_review(record_id, "CLEAR", "甲", "说明", 0)


def test_export_preserves_provenance_and_explicit_missing_images(
    tmp_path: Path,
) -> None:
    root, bundle = tmp_path / "data", tmp_path / "samples.zip"
    with InspectionStore(root) as store:
        item = seed(store, "NO_DETECTION")
        store.add_review(item["id"], "CONFIRMED", "=reviewer", "+人工发现细小裂缝", 0)
        missing = seed(store, original=None)
        store.add_review(missing["id"], "CLEAR", "甲", "已查全图", 0)
        seed(store)  # Unreviewed data must not enter the bundle.
        report = tmp_path / "report.csv"
        assert store.export_csv(report, review_decision="CONFIRMED") == 1
    with report.open(encoding="utf-8-sig", newline="") as file:
        row = next(csv.DictReader(file))
        assert row["reviewer"] == "'=reviewer" and row["review_note"].startswith("'+")
    summary = export_review_bundle(root, bundle)
    assert (summary["records"], summary["images"], summary["missing_images"]) == (
        2,
        1,
        1,
    )
    with ZipFile(bundle) as archive:
        records = [
            json.loads(archive.read(name))
            for name in archive.namelist()
            if name.endswith("record.json")
        ]
        included = next(r for r in records if r["image_status"] == "included")
        assert included["category"] == "possible_miss"
        assert included["run"]["model_sha256"] == "model-hash"
        assert included["source_sha256"] == "source-hash"
        assert included["exported_image_sha256"] == hashlib.sha256(b"image").hexdigest()
        assert included["review_history"][0]["reviewer"] == "=reviewer"
        assert not included["predictions_are_ground_truth"]
        assert not any(name.endswith(".txt") for name in archive.namelist())


def test_empty_filter_does_not_replace_existing_export(tmp_path: Path) -> None:
    bundle = tmp_path / "samples.zip"
    bundle.write_bytes(b"previous export")
    with InspectionStore(tmp_path / "data") as store:
        seed(store)
    with pytest.raises(ValueError, match="没有已复核"):
        export_review_bundle(tmp_path / "data", bundle)
    assert bundle.read_bytes() == b"previous export"
    assert not list(tmp_path.glob(".*.tmp"))


def test_export_rejects_original_path_outside_data_root(tmp_path: Path) -> None:
    root = tmp_path / "data"
    with InspectionStore(root) as store:
        item = seed(store)
        store.add_review(item["id"], "CLEAR", "甲", "已查全图", 0)
        with store.connection:
            store.connection.execute(
                "UPDATE inspections SET original_path='../outside.png'"
            )
    with pytest.raises(ValueError, match="超出"):
        export_review_bundle(root, tmp_path / "samples.zip")
    assert not (tmp_path / "samples.zip").exists()


def test_review_categories_are_image_level() -> None:
    assert review_category("DEFECT", "CLEAR") == "possible_false_alarm"
    assert review_category("DEFECT", "UNCERTAIN") == "needs_recheck"
    assert review_category("DEFECT", "CONFIRMED") == "human_machine_agreement"
