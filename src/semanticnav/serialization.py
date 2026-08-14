"""Run-directory creation and structured result serialization."""

import csv
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from semanticnav.models import FrameResult, PlannedPath


def create_run_directory(root: str | Path) -> tuple[str, Path]:
    output_root = Path(root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
    run_dir = output_root / run_id
    run_dir.mkdir(exist_ok=False)
    return run_id, run_dir


def write_json(path: str | Path, payload: object) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.{uuid4().hex}.tmp"
    )
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, default=str)
            file.write("\n")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_results_json(
    path: str | Path,
    metadata: dict[str, object],
    frames: list[FrameResult],
) -> None:
    write_json(
        path,
        {
            "metadata": metadata,
            "frames": [frame.model_dump(mode="json") for frame in frames],
        },
    )


def write_metrics_csv(path: str | Path, frames: list[FrameResult]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_index",
        "timestamp_s",
        "inference_ms",
        "total_ms",
        "object_count",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for frame in frames:
            writer.writerow(
                {
                    "frame_index": frame.frame_index,
                    "timestamp_s": frame.timestamp_s,
                    "inference_ms": frame.inference_ms,
                    "total_ms": frame.total_ms,
                    "object_count": len(frame.objects),
                }
            )


def write_path_csv(path: str | Path, planned_path: PlannedPath) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["step", "row", "column"])
        writer.writeheader()
        for step, (row, column) in enumerate(planned_path.cells):
            writer.writerow({"step": step, "row": row, "column": column})
