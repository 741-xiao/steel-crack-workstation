"""Responsive offline UI; inference and persistence run in a worker thread."""

# Qt entry points report operational failures to users instead of aborting Qt.
# ruff: noqa: BLE001

import json
import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from threading import Event
from time import monotonic
from typing import cast

# isort: off
import torch  # noqa: F401  # Windows: load PyTorch DLLs before importing PyQt5.
import cv2
import numpy as np
import PyQt5

# isort: on
from PyQt5.QtCore import QCoreApplication, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QCloseEvent, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .cli import DEFAULT_CONFIG, configure_logging
from .config import InspectionConfig
from .review import export_review_bundle
from .service import Image, InspectionEvent, RunSummary, inspect_batch
from .storage import REVIEW_LABELS, InspectionStore

STATUS_LABELS = {
    "DEFECT": "疑似裂缝",
    "NO_DETECTION": "未检出（不代表合格）",
    "ERROR": "处理失败",
    "RUNNING": "进行中 / 中断后未结束",
    "FAILED": "失败",
    "CANCELLED": "已停止",
    "COMPLETED": "已完成",
    "COMPLETED_WITH_ERRORS": "已完成（存在失败项）",
}


class InspectionWorker(QThread):
    """Publish previews at no more than five per second."""

    preview = pyqtSignal(object, int, int, int)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, config: InspectionConfig, source: Path, batch: str) -> None:
        super().__init__()
        self.config, self.source, self.batch = config, source, batch
        self.stop_event = Event()

    def run(self) -> None:
        processed = defects = errors = 0
        last_preview = 0.0

        def report(event: InspectionEvent) -> None:
            nonlocal processed, defects, errors, last_preview
            processed += event.record["status"] != "ERROR"
            defects += event.record["status"] == "DEFECT"
            errors += event.record["status"] == "ERROR"
            if monotonic() - last_preview >= 0.2:
                self.preview.emit(event, processed, defects, errors)
                last_preview = monotonic()

        try:
            self.completed.emit(
                inspect_batch(
                    self.config, self.source, self.batch, self.stop_event, report
                )
            )
        except Exception as error:
            self.failed.emit(str(error))


class CrackApp(QMainWindow):
    """Inspect batches, review recent outcomes and export persisted evidence."""

    def __init__(self, config_path: Path = DEFAULT_CONFIG) -> None:
        super().__init__()
        self.config = InspectionConfig.load(config_path)
        configure_logging(Path(self.config.output))
        self.worker: InspectionWorker | None = None
        self.close_pending = False
        self.setWindowTitle("工业裂缝检测离线版 V1.0")
        self.selected_record: dict[str, object] | None = None
        self.resize(1260, 850)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.addWidget(
            QLabel("离线检测与结果追溯  |  未检出不代表产品合格，检测结果需复核")
        )
        form = QFormLayout()
        self.weights = QLineEdit(self.config.weights)
        self.source = QLineEdit()
        self.batch = QLineEdit(datetime.now().strftime("批次_%Y%m%d_%H%M%S"))
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.01, 1.0)
        self.confidence.setSingleStep(0.05)
        self.confidence.setValue(self.config.confidence)
        model_row = QHBoxLayout()
        model_row.addWidget(self.weights)
        self.model_button = QPushButton("选择模型")
        self.model_button.clicked.connect(self.choose_model)
        model_row.addWidget(self.model_button)
        source_row = QHBoxLayout()
        source_row.addWidget(self.source)
        self.file_button = QPushButton("选择图片 / 视频")
        self.folder_button = QPushButton("选择文件夹")
        self.file_button.clicked.connect(self.choose_file)
        self.folder_button.clicked.connect(self.choose_folder)
        source_row.addWidget(self.file_button)
        source_row.addWidget(self.folder_button)
        form.addRow("本地模型", model_row)
        form.addRow("检测输入", source_row)
        form.addRow("批次 / 工件编号", self.batch)
        form.addRow("置信度阈值", self.confidence)
        layout.addLayout(form)
        controls = QHBoxLayout()
        self.start_button = QPushButton("开始检测")
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_inspection)
        self.stop_button.clicked.connect(self.stop_inspection)
        self.status = QLabel("就绪")
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.status, 1)
        layout.addLayout(controls)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.image = QLabel("选择图片、视频或文件夹后开始检测")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(480, 320)
        self.image.setStyleSheet(
            "background:#18212b; color:#e0e6ed; border:1px solid #405269"
        )
        splitter.addWidget(self.image)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlaceholderText("点击历史记录，查看检测详情、坐标与错误原因")
        splitter.addWidget(self.details)
        splitter.setSizes([850, 350])
        layout.addWidget(splitter, 1)
        history_controls = QHBoxLayout()
        history_controls.addWidget(QLabel("历史记录（最近 200 条）"))
        self.run_filter = QComboBox()
        self.run_filter.setMinimumWidth(180)
        self.run_filter.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.run_filter.currentIndexChanged.connect(self.refresh_records)
        history_controls.addWidget(self.run_filter, 1)
        refresh = QPushButton("刷新历史")
        refresh.clicked.connect(self.refresh_history)
        self.export_button = QPushButton("导出 CSV")
        self.export_button.clicked.connect(self.export_csv)
        history_controls.addWidget(refresh)
        history_controls.addWidget(self.export_button)
        layout.addLayout(history_controls)
        review_controls = QHBoxLayout()
        review_controls.addWidget(QLabel("复核筛选"))
        self.review_filter = QComboBox()
        self.review_filter.addItem("全部复核状态", None)
        for decision, label in REVIEW_LABELS.items():
            self.review_filter.addItem(label, decision)
        self.review_filter.currentIndexChanged.connect(self.refresh_records)
        review_controls.addWidget(self.review_filter)
        self.original_view = QCheckBox("查看原图（便于复核）")
        self.original_view.toggled.connect(self.show_record)
        review_controls.addWidget(self.original_view)
        self.bundle_button = QPushButton("导出复核样本 ZIP")
        self.bundle_button.clicked.connect(self.export_bundle)
        review_controls.addWidget(self.bundle_button)
        review_controls.addStretch()
        layout.addLayout(review_controls)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            [
                "时间 (UTC)",
                "批次",
                "文件 / 帧",
                "机器结果",
                "预测框数",
                "推理耗时 ms",
                "人工复核",
            ]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = self.table.horizontalHeader()
        assert header is not None
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self.show_record)
        self.table.setMinimumHeight(220)
        layout.addWidget(self.table)
        review_form = QFormLayout()
        self.reviewer = QLineEdit()
        self.reviewer.setMaxLength(80)
        self.reviewer.setPlaceholderText("填写实际复核人姓名（手工记录）")
        self.review_decision = QComboBox()
        self.review_decision.addItem("请选择整张图片的复核结论", None)
        for decision, label in REVIEW_LABELS.items():
            if decision != "PENDING":
                self.review_decision.addItem(label, decision)
        reviewer_row = QHBoxLayout()
        reviewer_row.addWidget(self.reviewer)
        reviewer_row.addWidget(self.review_decision)
        self.review_save = QPushButton("保存人工复核")
        self.review_save.setEnabled(False)
        self.review_save.clicked.connect(self.save_review)
        reviewer_row.addWidget(self.review_save)
        review_form.addRow("人工复核", reviewer_row)
        self.review_note = QLineEdit()
        self.review_note.setMaxLength(2000)
        self.review_note.setPlaceholderText(
            "必填：复核依据；有裂缝不代表所有预测框正确，无裂缝不等于工件合格"
        )
        review_form.addRow("复核依据", self.review_note)
        layout.addLayout(review_form)
        layout.addWidget(
            QLabel(
                f"结果目录：{self.config.output}  |  保存策略：{self.config.save_images}"
            )
        )
        self.refresh_history()
        self.timer = QTimer(self)
        self.timer.setInterval(1500)
        self.timer.timeout.connect(self.refresh_records)

    def choose_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择可信的本地裂缝模型", "", "模型 (*.pt *.onnx)"
        )
        if path:
            self.weights.setText(path)

    def choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片或视频",
            "",
            "图片或视频 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.mp4 *.avi *.mov *.mkv *.m4v)",
        )
        if path:
            self.source.setText(path)

    def choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择批量检测目录")
        if path:
            self.source.setText(path)

    def start_inspection(self) -> None:
        if self.worker is not None:
            return
        try:
            if not self.source.text().strip() or not Path(self.source.text()).exists():
                raise ValueError("请选择存在的输入文件或文件夹")
            if not self.batch.text().strip():
                raise ValueError("请输入批次或工件编号")
            if not Path(self.weights.text()).is_file():
                raise ValueError("请选择存在的本地模型文件")
            config = replace(
                self.config,
                weights=str(Path(self.weights.text()).resolve()),
                confidence=self.confidence.value(),
            )
            self.worker = InspectionWorker(
                config, Path(self.source.text()), self.batch.text().strip()
            )
            self.worker.preview.connect(self.update_preview)
            self.worker.completed.connect(self.on_completed)
            self.worker.failed.connect(self.on_failed)
            self.worker.finished.connect(self.on_finished)
            self.set_busy(True)
            self.status.setText("正在加载模型并准备检测…")
            self.worker.start()
            self.timer.start()
        except Exception as error:
            QMessageBox.warning(self, "无法开始", str(error))

    def set_busy(self, busy: bool) -> None:
        for widget in (
            self.weights,
            self.source,
            self.batch,
            self.confidence,
            self.model_button,
            self.file_button,
            self.folder_button,
            self.start_button,
            self.export_button,
            self.bundle_button,
            self.reviewer,
            self.review_decision,
            self.review_note,
            self.table,
        ):
            widget.setEnabled(not busy)
        self.review_save.setEnabled(
            not busy
            and self.selected_record is not None
            and self.selected_record["status"] != "ERROR"
        )
        self.stop_button.setEnabled(busy)

    def stop_inspection(self) -> None:
        if self.worker is not None:
            self.worker.stop_event.set()
            self.status.setText("正在停止，等待当前帧和结果保存完成…")
            self.stop_button.setEnabled(False)

    def update_preview(
        self, event: InspectionEvent, processed: int, defects: int, errors: int
    ) -> None:
        self.status.setText(
            f"已检测 {processed} 张/帧 · 疑似裂缝 {defects} 张/帧 · 失败 {errors}"
        )
        if event.preview is not None:
            self.show_image(event.preview)

    def show_image(self, image: Image) -> None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        qimage = QImage(
            rgb.tobytes(), width, height, int(rgb.strides[0]), QImage.Format_RGB888
        ).copy()
        self.image.setPixmap(
            QPixmap.fromImage(qimage).scaled(
                self.image.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def on_completed(self, summary: RunSummary) -> None:
        self.status.setText(
            f"{STATUS_LABELS[summary.status]} · {summary.processed} 张/帧 · 疑似裂缝 {summary.defects} · 失败 {summary.errors}"
        )

    def on_failed(self, error: str) -> None:
        self.status.setText(f"检测失败：{error}")
        if not self.close_pending:
            QMessageBox.critical(self, "检测已停止", error)

    def on_finished(self) -> None:
        assert self.worker is not None
        self.worker.deleteLater()
        self.worker = None
        self.timer.stop()
        self.set_busy(False)
        self.refresh_history()
        if self.close_pending:
            self.close()

    def refresh_history(self) -> None:
        try:
            selected = self.run_filter.currentData()
            with InspectionStore(Path(self.config.output)) as store:
                runs = store.runs()
            self.run_filter.blockSignals(True)
            self.run_filter.clear()
            self.run_filter.addItem("全部批次", None)
            for run in runs:
                label = f"{run['batch']} | {run['started_at']} | {STATUS_LABELS.get(str(run['status']), str(run['status']))} | {str(run['id'])[:8]}"
                self.run_filter.addItem(label, run["id"])
            self.run_filter.setCurrentIndex(max(0, self.run_filter.findData(selected)))
            self.run_filter.blockSignals(False)
            self.refresh_records()
        except Exception as error:
            self.status.setText(f"历史读取失败：{error}")

    def refresh_records(self) -> None:
        try:
            with InspectionStore(Path(self.config.output)) as store:
                records = store.history(
                    self.run_filter.currentData(),
                    review_decision=self.review_filter.currentData(),
                )
            self.selected_record = None
            self.review_save.setEnabled(False)
            self.review_decision.setCurrentIndex(0)
            self.review_note.clear()
            self.details.clear()
            if self.worker is None:
                self.image.clear()
                self.image.setText("点击历史记录查看图片并复核")
            self.table.setRowCount(0)
            for row, record in enumerate(records):
                self.table.insertRow(row)
                frame = (
                    f" / 帧 {record['frame_index']}"
                    if record["frame_index"] is not None
                    else ""
                )
                values = [
                    record["created_at"],
                    record["batch"],
                    Path(str(record["source"])).name + frame,
                    STATUS_LABELS[str(record["status"])],
                    len(json.loads(str(record["detections_json"]))),
                    record["elapsed_ms"],
                    REVIEW_LABELS[str(record["review_decision"])],
                ]
                for column, value in enumerate(values):
                    cell = QTableWidgetItem(str(value))
                    cell.setData(Qt.ItemDataRole.UserRole, record)
                    self.table.setItem(row, column, cell)
        except Exception as error:
            self.status.setText(f"历史读取失败：{error}")

    def show_record(self) -> None:
        selected = self.table.selectedItems()
        if not selected:
            return
        record = selected[0].data(Qt.ItemDataRole.UserRole)
        changed = (
            self.selected_record is None or self.selected_record["id"] != record["id"]
        )
        self.selected_record = record
        self.review_save.setEnabled(self.worker is None and record["status"] != "ERROR")
        if changed:
            self.review_decision.setCurrentIndex(
                max(0, self.review_decision.findData(record["review_decision"]))
            )
            self.review_note.setText(str(record["review_note"]))
        with InspectionStore(Path(self.config.output)) as store:
            run = store.connection.execute(
                "SELECT * FROM runs WHERE id=?", (record["run_id"],)
            ).fetchone()
            revisions = store.review_history(str(record["id"]))
        details = dict(record)
        details["review_history"] = revisions
        details["detections"] = json.loads(details.pop("detections_json"))
        if run is not None:
            details["model_sha256"] = run["model_sha256"]
            details["settings"] = json.loads(run["config_json"])
        self.details.setPlainText(json.dumps(details, ensure_ascii=False, indent=2))
        path = (
            record.get("original_path")
            if self.original_view.isChecked()
            else record.get("annotated_path") or record.get("original_path")
        )
        try:
            image = (
                cv2.imdecode(
                    np.fromfile(Path(self.config.output) / path, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if path
                else None
            )
            if image is not None:
                self.show_image(cast(Image, image))
            else:
                self.image.clear()
                self.image.setText("该记录没有保存图片；请查看详情")
        except (OSError, cv2.error) as error:
            self.image.clear()
            self.image.setText(f"历史图片无法读取：{error}")

    def export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "导出当前筛选的全部记录", "检测记录.csv", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            with InspectionStore(Path(self.config.output)) as store:
                count = store.export_csv(
                    Path(path),
                    self.run_filter.currentData(),
                    self.review_filter.currentData(),
                )
            self.status.setText(f"已导出 {count} 条记录：{path}")
        except Exception as error:
            QMessageBox.critical(self, "导出失败", str(error))

    def save_review(self) -> None:
        """Append a human decision against the exact version displayed to the user."""
        record = self.selected_record
        if record is None or self.worker is not None:
            return
        try:
            with InspectionStore(Path(self.config.output)) as store:
                store.add_review(
                    str(record["id"]),
                    self.review_decision.currentData(),
                    self.reviewer.text(),
                    self.review_note.text(),
                    int(record["review_version"]),
                )
            self.refresh_records()
            self.status.setText("人工复核已保存；机器预测保留，可在详情中查看历次复核")
        except Exception as error:
            QMessageBox.warning(self, "复核未保存", str(error))

    def export_bundle(self) -> None:
        """Export reviewed evidence under the current filters for later annotation."""
        path, _ = QFileDialog.getSaveFileName(
            self, "导出当前筛选的已复核样本", "复核样本.zip", "ZIP (*.zip)"
        )
        if not path:
            return
        try:
            summary = export_review_bundle(
                Path(self.config.output),
                Path(path),
                self.run_filter.currentData(),
                self.review_filter.currentData(),
            )
            self.status.setText(
                f"已导出 {summary['records']} 条复核，{summary['images']} 张原图；"
                f"缺少原图 {summary['missing_images']} 条。需人工标注后才能训练。"
            )
        except Exception as error:
            QMessageBox.warning(self, "样本导出失败", str(error))

    def closeEvent(self, event: QCloseEvent | None) -> None:
        if event is None:
            return
        if self.worker is not None:
            self.close_pending = True
            self.stop_inspection()
            event.ignore()
        else:
            event.accept()


def main() -> int:
    """Launch the workstation using settings adjacent to the application."""
    plugin_dir = Path(PyQt5.__file__).resolve().parent / "Qt5" / "plugins"
    platform_plugin = plugin_dir / "platforms" / "qwindows.dll"
    if not platform_plugin.is_file():
        QMessageBox.critical(
            None,
            "工作站启动失败",
            f"未找到 Qt Windows 界面插件：{platform_plugin}\n请修复当前 Python 环境中的 PyQt5 安装。",
        )
        return 1
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platform_plugin.parent)
    QCoreApplication.addLibraryPath(str(plugin_dir))
    app = QApplication(sys.argv)
    try:
        window = CrackApp()
    except Exception as error:
        QMessageBox.critical(None, "工作站启动失败", str(error))
        return 1
    window.show()
    return int(app.exec_())
