"""Export human-reviewed evidence for later annotation, without inventing labels."""

import hashlib
import json
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from . import __version__
from .storage import InspectionStore, utc_now


def review_category(machine: str, decision: str) -> str:
    """Classify image-level disagreement; this is not a box-level accuracy metric."""
    if decision == "UNCERTAIN":
        return "needs_recheck"
    if machine == "NO_DETECTION" and decision == "CONFIRMED":
        return "possible_miss"
    if machine == "DEFECT" and decision == "CLEAR":
        return "possible_false_alarm"
    return "human_machine_agreement"


def export_review_bundle(
    root: Path,
    path: Path,
    run_id: str | None = None,
    review_decision: str | None = None,
) -> dict[str, object]:
    """Export a consistent snapshot of reviewed records into a local ZIP.

    Args:
        root: Inspection data directory containing the SQLite database.
        path: Destination ZIP path; parent directory must exist.
        run_id: Optional batch run filter.
        review_decision: Optional human decision filter matching the history UI.

    Returns:
        Counts of reviewed records, copied images and records without images.

    Raises:
        ValueError: No reviewed records match, or a stored path leaves root.
        OSError: Evidence cannot be read or the destination cannot be written.

    Missing originals are explicitly recorded. Predicted boxes remain suggestions;
    no YOLO label files are generated. Exported PNG hashes describe the copied
    evidence, while source_sha256 describes the original detection input.
    """
    root, path = root.resolve(), path.resolve()
    if path.suffix.lower() != ".zip":
        raise ValueError("样本包必须使用 .zip 扩展名")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    summary: dict[str, object] = {
        "format_version": 1,
        "app_version": __version__,
        "created_at": utc_now(),
        "run_id": run_id,
        "review_filter": review_decision,
        "records": 0,
        "images": 0,
        "missing_images": 0,
        "purpose": "待人工标注样本；不能直接用于训练",
    }
    counts = {"records": 0, "images": 0, "missing_images": 0}
    try:
        with (
            InspectionStore(root) as store,
            ZipFile(temporary, "x", ZIP_DEFLATED) as archive,
        ):
            # Hold a read snapshot so latest decisions and revision lists agree.
            store.connection.execute("BEGIN")
            query, params = store.result_query(run_id, review_decision)
            for row in store.connection.execute(query, params):
                record = dict(row)
                if record["review_decision"] == "PENDING":
                    continue
                counts["records"] += 1
                record["category"] = review_category(
                    record["status"], record["review_decision"]
                )
                record["review_history"] = store.review_history(record["id"])
                run = store.connection.execute(
                    "SELECT * FROM runs WHERE id=?", (record["run_id"],)
                ).fetchone()
                record["run"] = dict(run)
                record["predictions_are_ground_truth"] = False
                # Use numeric review ID for archive paths, never user-entered text.
                prefix = f"samples/review_{int(record['review_version'])}"
                original = record["original_path"]
                evidence = (root / original).resolve() if original else None
                if evidence is not None and not evidence.is_relative_to(root):
                    raise ValueError("原图路径超出结果目录，导出已停止")
                if evidence is None or not evidence.is_file():
                    record["image_status"] = "missing_original"
                    counts["missing_images"] += 1
                else:
                    digest = hashlib.sha256()
                    with (
                        evidence.open("rb") as source,
                        archive.open(f"{prefix}/original.png", "w") as target,
                    ):
                        while chunk := source.read(1024 * 1024):
                            digest.update(chunk)
                            target.write(chunk)
                    record["image_status"] = "included"
                    record["exported_image_sha256"] = digest.hexdigest()
                    counts["images"] += 1
                archive.writestr(
                    f"{prefix}/record.json",
                    json.dumps(record, ensure_ascii=False, indent=2),
                )
            if not counts["records"]:
                raise ValueError("当前筛选下没有已复核记录，请先保存人工复核")
            summary.update(counts)
            archive.writestr(
                "summary.json", json.dumps(summary, ensure_ascii=False, indent=2)
            )
            archive.writestr(
                "README.md",
                (
                    "# 人工复核样本包\n\n"
                    "这是待标注资料，不是可直接训练的数据集。每个 samples 子目录包含复核记录，"
                    "原图存在时同时包含 original.png。请先查看 summary.json 的 missing_images。\n\n"
                    "record.json 保留模型预测、参数、模型/源文件指纹和全部复核历史。"
                    "有裂缝/无裂缝结论针对整张图，不表示每个框正确或全部裂缝已标完。"
                    "请使用标注工具检查全图并补画/修正框，完成独立复查后再用于训练。"
                    "同一工件、连续视频帧应划入同一数据子集，避免训练与测试泄漏。\n\n"
                    "category 是人工与机器的图像级分歧分类，不是精确率/召回率。"
                    "仅针对 ROI 的推理不能代表全图覆盖。复核人姓名为手工输入，未经身份认证。\n"
                ),
            )
        temporary.replace(path)
        return summary
    finally:
        temporary.unlink(missing_ok=True)
