"""The workstation must start and display committed history without a monitor."""

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from industrial_crack.storage import InspectionStore


@pytest.mark.skipif(
    os.environ.get("RUN_GUI_TESTS") != "1",
    reason="Qt GUI needs an interactive desktop session",
)
def test_workstation_shows_history_and_details(tmp_path: Path) -> None:
    # isort: off
    from industrial_crack.gui import CrackApp
    from PyQt5.QtWidgets import QApplication
    # isort: on

    config_path = tmp_path / "inspection_config.json"
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"model")
    config_path.write_text(
        json.dumps({"weights": "best.pt", "output": "results", "min_free_mb": 1}),
        encoding="utf-8",
    )
    app = QApplication.instance() or QApplication([])
    with InspectionStore(tmp_path / "results") as store:
        run_id = store.begin("test-batch", "abc", {"confidence": 0.25}, {"app": "test"})
        store.record(run_id, "example.png", None, "hash", "NO_DETECTION", [], 10)
        store.finish(run_id, "COMPLETED")
    window = CrackApp(config_path)
    assert window.table.rowCount() == 1
    window.table.selectRow(0)
    app.processEvents()
    assert "test-batch" in window.details.toPlainText()
    assert "model_sha256" in window.details.toPlainText()
    window.reviewer.setText("测试复核人")
    window.review_decision.setCurrentIndex(window.review_decision.findData("CONFIRMED"))
    window.review_note.setText("原图存在模型漏掉的裂缝，需补充标注")
    window.original_view.setChecked(True)
    assert window.review_note.text() == "原图存在模型漏掉的裂缝，需补充标注"
    assert window.review_decision.currentData() == "CONFIRMED"
    window.review_save.click()
    window.table.selectRow(0)
    app.processEvents()
    assert "测试复核人" in window.details.toPlainText()
    assert "review_history" in window.details.toPlainText()
    with InspectionStore(tmp_path / "results") as store:
        assert store.history()[0]["status"] == "NO_DETECTION"
        assert store.history()[0]["review_decision"] == "CONFIRMED"
    window.review_filter.setCurrentIndex(window.review_filter.findData("PENDING"))
    assert window.table.rowCount() == 0
    window.close()
