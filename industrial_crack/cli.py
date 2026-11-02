"""Command-line inspection, history lookup and traceability export."""

import argparse
import json
import logging
import signal
from dataclasses import asdict, replace
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Event

from .config import InspectionConfig
from .review import export_review_bundle
from .service import InspectionEvent, inspect_batch
from .storage import REVIEW_LABELS, InspectionStore

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "inspection_config.json"


def configure_logging(root: Path) -> None:
    """Keep bounded application logs under the configured result directory."""
    root.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        root / "inspection.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger = logging.getLogger("industrial_crack")
    for previous in logger.handlers[:]:
        previous.close()
        logger.removeHandler(previous)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def main(argv: list[str] | None = None) -> int:
    """Run a local batch or query its durable history."""
    parser = argparse.ArgumentParser(description="钢结构裂缝离线检测与追溯")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect", help="检测图片、视频或文件夹")
    inspect.add_argument("--source", type=Path, required=True)
    inspect.add_argument("--batch", required=True, help="工件/批次名称")
    inspect.add_argument("--weights", type=Path)
    inspect.add_argument("--conf", type=float)
    subparsers.add_parser("history", help="查看近期批次及参数")
    export = subparsers.add_parser("export", help="导出 CSV")
    export.add_argument("--file", type=Path, required=True)
    export.add_argument("--run-id", help="省略时导出全部记录")
    export.add_argument("--review-state", choices=list(REVIEW_LABELS))
    samples = subparsers.add_parser("export-samples", help="导出已复核的待标注样本 ZIP")
    samples.add_argument("--file", type=Path, required=True)
    samples.add_argument("--run-id")
    samples.add_argument("--review-state", choices=list(REVIEW_LABELS))
    args = parser.parse_args(argv)
    try:
        config = InspectionConfig.load(args.config)
        configure_logging(Path(config.output))
        if args.command == "inspect":
            if args.weights:
                config = replace(config, weights=str(args.weights.resolve()))
            if args.conf is not None:
                config = replace(config, confidence=args.conf)
            stop = Event()
            previous = signal.signal(signal.SIGINT, lambda *_: stop.set())
            try:

                def progress(event: InspectionEvent) -> None:
                    print(
                        f"{event.record['status']} {event.record['source']} "
                        f"frame={event.record['frame_index']}",
                        flush=True,
                    )

                summary = inspect_batch(config, args.source, args.batch, stop, progress)
            finally:
                signal.signal(signal.SIGINT, previous)
            print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
            return {"COMPLETED": 0, "CANCELLED": 130}.get(summary.status, 2)
        if args.command == "export-samples":
            summary = export_review_bundle(
                Path(config.output), args.file, args.run_id, args.review_state
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        with InspectionStore(Path(config.output)) as store:
            if args.command == "history":
                print(json.dumps(store.runs(), ensure_ascii=False, indent=2))
            else:
                count = store.export_csv(args.file, args.run_id, args.review_state)
                print(f"已导出 {count} 条记录：{args.file.resolve()}")
        return 0
    except Exception as error:
        logging.getLogger(__name__).exception("操作失败")
        print(f"操作失败：{error}")
        return 1
