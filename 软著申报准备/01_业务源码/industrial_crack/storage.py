"""SQLite run history, atomic image files and spreadsheet-safe CSV exports."""

import csv
import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self
from uuid import uuid4

REVIEW_LABELS = {
    "PENDING": "未复核",
    "CONFIRMED": "人工确认有裂缝",
    "CLEAR": "人工确认无裂缝",
    "UNCERTAIN": "待进一步复查",
}

# Keep machine outcomes immutable; the latest human opinion is a separate view.
HISTORY_QUERY = """
    SELECT inspections.*, runs.batch,
        COALESCE(reviews.id, 0) AS review_version,
        COALESCE(reviews.decision, 'PENDING') AS review_decision,
        COALESCE(reviews.reviewer, '') AS reviewer,
        COALESCE(reviews.note, '') AS review_note,
        COALESCE(reviews.created_at, '') AS reviewed_at
    FROM inspections JOIN runs ON runs.id=inspections.run_id
    LEFT JOIN reviews ON reviews.id=(
        SELECT MAX(r.id) FROM reviews r WHERE r.inspection_id=inspections.id
    )
"""


def utc_now() -> str:
    """Use unambiguous UTC timestamps in persisted records."""
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def atomic_write(path: Path, data: bytes) -> None:
    """Commit a file using a same-directory atomic rename."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class InspectionStore:
    """One connection per worker; images are written before their database row."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.root / "inspections.sqlite3", timeout=10)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, batch TEXT NOT NULL, started_at TEXT NOT NULL,
                finished_at TEXT, status TEXT NOT NULL, model_sha256 TEXT NOT NULL,
                config_json TEXT NOT NULL, environment_json TEXT NOT NULL,
                error TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS inspections (
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
                created_at TEXT NOT NULL, source TEXT NOT NULL, frame_index INTEGER,
                source_sha256 TEXT NOT NULL, status TEXT NOT NULL,
                detections_json TEXT NOT NULL, elapsed_ms REAL NOT NULL,
                original_path TEXT, annotated_path TEXT, error TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS inspections_run ON inspections(run_id);
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_id TEXT NOT NULL REFERENCES inspections(id),
                created_at TEXT NOT NULL, reviewer TEXT NOT NULL,
                decision TEXT NOT NULL CHECK(decision IN ('CONFIRMED','CLEAR','UNCERTAIN')),
                note TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS reviews_inspection ON reviews(inspection_id, id);
        """)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.connection.close()

    def begin(
        self,
        batch: str,
        model_sha256: str,
        config: dict[str, object],
        environment: dict[str, str],
    ) -> str:
        """Persist provenance before the first inference."""
        run_id = uuid4().hex
        (self.root / run_id).mkdir()
        with self.connection:
            self.connection.execute(
                "INSERT INTO runs VALUES (?,?,?,NULL,'RUNNING',?,?,?,'')",
                (
                    run_id,
                    batch,
                    utc_now(),
                    model_sha256,
                    json.dumps(config, ensure_ascii=False),
                    json.dumps(environment),
                ),
            )
        return run_id

    def finish(self, run_id: str, status: str, error: str = "") -> None:
        """Close a run, preserving ERROR/CANCELLED independently of detections."""
        with self.connection:
            self.connection.execute(
                "UPDATE runs SET finished_at=?,status=?,error=? WHERE id=?",
                (utc_now(), status, error, run_id),
            )

    def record(
        self,
        run_id: str,
        source: str,
        frame_index: int | None,
        source_sha256: str,
        status: str,
        detections: list[dict[str, object]],
        elapsed_ms: float,
        original: bytes | None = None,
        annotated: bytes | None = None,
        error: str = "",
    ) -> dict[str, object]:
        """Store one outcome; failures propagate so a run cannot silently succeed."""
        inspection_id = uuid4().hex
        paths: list[str | None] = [None, None]
        created: list[Path] = []
        try:
            for i, (suffix, data) in enumerate(
                (("original", original), ("annotated", annotated))
            ):
                if data is not None:
                    relative = Path(run_id) / f"{inspection_id}_{suffix}.png"
                    destination = self.root / relative
                    atomic_write(destination, data)
                    created.append(destination)
                    paths[i] = relative.as_posix()
            row: dict[str, object] = {
                "id": inspection_id,
                "run_id": run_id,
                "created_at": utc_now(),
                "source": source,
                "frame_index": frame_index,
                "source_sha256": source_sha256,
                "status": status,
                "detections_json": json.dumps(
                    detections, ensure_ascii=False, allow_nan=False
                ),
                "elapsed_ms": round(elapsed_ms, 3),
                "original_path": paths[0],
                "annotated_path": paths[1],
                "error": error,
            }
            with self.connection:
                self.connection.execute(
                    "INSERT INTO inspections VALUES (:id,:run_id,:created_at,:source,"
                    ":frame_index,:source_sha256,:status,:detections_json,:elapsed_ms,"
                    ":original_path,:annotated_path,:error)",
                    row,
                )
            return row
        except Exception:
            for path in created:
                path.unlink(missing_ok=True)
            raise

    def history(
        self,
        run_id: str | None = None,
        limit: int = 200,
        review_decision: str | None = None,
    ) -> list[dict[str, object]]:
        """Return recent outcomes, optionally restricted to a batch run."""
        query, params = self.result_query(run_id, review_decision)
        rows = self.connection.execute(
            query + " ORDER BY inspections.rowid DESC LIMIT ?", (*params, limit)
        )
        return [dict(row) for row in rows]

    def result_query(
        self, run_id: str | None = None, review_decision: str | None = None
    ) -> tuple[str, tuple[object, ...]]:
        """Build the shared history/export filter; reject unknown review states."""
        clauses, params = [], []
        if run_id:
            clauses.append("inspections.run_id=?")
            params.append(run_id)
        if review_decision is not None:
            if review_decision not in REVIEW_LABELS:
                raise ValueError("未知复核状态")
            clauses.append("COALESCE(reviews.decision, 'PENDING')=?")
            params.append(review_decision)
        return HISTORY_QUERY + (
            " WHERE " + " AND ".join(clauses) if clauses else ""
        ), tuple(params)

    def review_history(self, inspection_id: str) -> list[dict[str, object]]:
        """Return every human revision in commit order, including superseded ones."""
        return [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM reviews WHERE inspection_id=? ORDER BY id",
                (inspection_id,),
            )
        ]

    def add_review(
        self,
        inspection_id: str,
        decision: str,
        reviewer: str,
        note: str,
        expected_version: int,
    ) -> int:
        """Append a review without changing predictions.

        Args:
            inspection_id: Persisted image/frame ID.
            decision: CONFIRMED, CLEAR or UNCERTAIN; applies to the entire image.
            reviewer: Operator-entered name (not an authenticated identity).
            note: Required explanation, up to 2000 characters.
            expected_version: Latest review ID seen by the caller, or zero.

        Returns:
            The new review ID.

        Raises:
            ValueError: Invalid fields, missing inspection or failed inspection.
            RuntimeError: Another reviewer has already changed this record.
        """
        if decision not in REVIEW_LABELS or decision == "PENDING":
            raise ValueError("请选择有效的复核结论")
        reviewer, note = reviewer.strip(), note.strip()
        if not reviewer or len(reviewer) > 80:
            raise ValueError("复核人必填，最多 80 字")
        if not note or len(note) > 2000:
            raise ValueError("复核依据必填，最多 2000 字")
        if type(expected_version) is not int or expected_version < 0:
            raise ValueError("复核版本无效")
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            record = self.connection.execute(
                "SELECT status FROM inspections WHERE id=?", (inspection_id,)
            ).fetchone()
            if record is None or record["status"] == "ERROR":
                raise ValueError("记录不存在或处理失败；请先重新检测")
            version = self.connection.execute(
                "SELECT COALESCE(MAX(id),0) FROM reviews WHERE inspection_id=?",
                (inspection_id,),
            ).fetchone()[0]
            if version != expected_version:
                raise RuntimeError("该记录已有新的复核，请刷新后查看再提交")
            cursor = self.connection.execute(
                "INSERT INTO reviews (inspection_id,created_at,reviewer,decision,note) VALUES (?,?,?,?,?)",
                (inspection_id, utc_now(), reviewer, decision, note),
            )
            return int(cursor.lastrowid)

    def runs(self, limit: int = 100) -> list[dict[str, object]]:
        """Include unfinished runs so interrupted jobs remain visible."""
        return [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM runs ORDER BY rowid DESC LIMIT ?", (limit,)
            )
        ]

    def export_csv(
        self,
        path: Path,
        run_id: str | None = None,
        review_decision: str | None = None,
    ) -> int:
        """Stream records into an atomic UTF-8 CSV; neutralize spreadsheet formulas."""
        query, params = self.result_query(run_id, review_decision)
        rows = self.connection.execute(query, params)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        count = 0
        try:
            with temporary.open("x", encoding="utf-8-sig", newline="") as handle:
                assert rows.description is not None
                writer = csv.writer(handle)
                writer.writerow([column[0] for column in rows.description])
                for row in rows:
                    writer.writerow(
                        [
                            (
                                "'" + value
                                if isinstance(value, str)
                                and value.lstrip().startswith(("=", "+", "-", "@"))
                                else value
                            )
                            for value in row
                        ]
                    )
                    count += 1
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return count
