import csv
from pathlib import Path

import pytest

from scripts.benchmark import (
    build_scenarios,
    summarize_run,
    write_benchmark_csv,
)


def test_summarize_run_preserves_measured_metrics() -> None:
    row = summarize_run(
        {
            "run_id": "run-1",
            "status": "completed",
            "frame_count": 30,
            "average_fps": 4.221,
            "average_inference_ms": 148.218,
            "p95_total_ms": 317.615,
            "planning_result": {"success": True},
        }
    )

    assert row == {
        "run_id": "run-1",
        "status": "completed",
        "frame_count": 30,
        "average_fps": pytest.approx(4.221),
        "average_inference_ms": pytest.approx(148.218),
        "p95_total_ms": pytest.approx(317.615),
        "path_success": True,
    }


def test_summarize_run_rejects_missing_metric() -> None:
    with pytest.raises(ValueError, match="average_fps"):
        summarize_run({"run_id": "run-1"})


def test_build_scenarios_uses_shorter_depth_run() -> None:
    scenarios = build_scenarios(max_frames=100)

    assert [scenario.name for scenario in scenarios] == ["E1", "E2", "E3", "E4"]
    assert [scenario.max_frames for scenario in scenarios] == [100, 100, 100, 20]
    assert scenarios[3].depth_enabled is True
    assert scenarios[3].depth_frame_interval == 5


def test_write_benchmark_csv_uses_stable_field_order(tmp_path: Path) -> None:
    output_path = tmp_path / "benchmark.csv"
    write_benchmark_csv(
        output_path,
        [
            {
                "scenario": "E1",
                "model": "yolo26n.pt",
                "image_size": 480,
                "confidence": 0.25,
                "depth_enabled": False,
                "depth_frame_interval": 5,
                "run_id": "run-1",
                "status": "completed",
                "frame_count": 30,
                "average_fps": 4.2,
                "average_inference_ms": 148.2,
                "p95_total_ms": 317.6,
                "path_success": True,
            }
        ],
    )

    with output_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert rows[0]["scenario"] == "E1"
    assert rows[0]["average_fps"] == "4.2"
    assert list(rows[0])[:4] == [
        "scenario",
        "model",
        "image_size",
        "confidence",
    ]
