import csv
import json
from pathlib import Path

from semanticnav.models import FrameResult
from semanticnav.serialization import (
    create_run_directory,
    write_metrics_csv,
    write_results_json,
)


def _empty_frame(frame_index: int = 0) -> FrameResult:
    return FrameResult(
        frame_index=frame_index,
        timestamp_s=frame_index / 10,
        inference_ms=10,
        total_ms=12,
        objects=[],
    )


def test_write_results_keeps_empty_objects(tmp_path: Path) -> None:
    path = tmp_path / "results.json"

    write_results_json(path, {"run_id": "run-1"}, [_empty_frame()])

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["metadata"]["run_id"] == "run-1"
    assert data["frames"][0]["objects"] == []


def test_create_run_directory_is_unique_for_immediate_calls(tmp_path: Path) -> None:
    first_id, first_dir = create_run_directory(tmp_path)
    second_id, second_dir = create_run_directory(tmp_path)

    assert first_id != second_id
    assert first_dir != second_dir
    assert first_dir.is_dir()
    assert second_dir.is_dir()


def test_write_metrics_csv_uses_fixed_columns(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"

    write_metrics_csv(path, [_empty_frame(frame_index=2)])

    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert list(rows[0]) == [
        "frame_index",
        "timestamp_s",
        "inference_ms",
        "total_ms",
        "object_count",
    ]
    assert rows[0]["frame_index"] == "2"
    assert rows[0]["object_count"] == "0"


def test_write_results_leaves_no_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "results.json"

    write_results_json(path, {}, [_empty_frame()])

    assert [item for item in tmp_path.iterdir() if item != path] == []
