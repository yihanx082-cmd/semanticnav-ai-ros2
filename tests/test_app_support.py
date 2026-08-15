import json
from pathlib import Path

import pytest

from semanticnav.app_support import (
    collect_artifacts,
    format_metrics,
    load_json,
    save_uploaded_video,
)
from semanticnav.models import PlannedPath, SemanticTask
from semanticnav.pipeline import RunSummary


def test_save_uploaded_video_strips_parent_directories(tmp_path: Path) -> None:
    saved = save_uploaded_video("../room.mp4", b"video", tmp_path)

    assert saved.parent == tmp_path
    assert saved.name.startswith("room-")
    assert saved.suffix == ".mp4"
    assert saved.read_bytes() == b"video"


@pytest.mark.parametrize("name", ["room.avi", "room.txt", "room"])
def test_save_uploaded_video_rejects_non_mp4(
    name: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="MP4"):
        save_uploaded_video(name, b"video", tmp_path)


def test_save_uploaded_video_rejects_empty_content(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="空"):
        save_uploaded_video("room.mp4", b"", tmp_path)


def test_collect_artifacts_returns_only_existing_allowlisted_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "results.json").write_text("{}", encoding="utf-8")
    (tmp_path / "metrics.csv").write_text("frame_index\n", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")

    artifacts = collect_artifacts(tmp_path)

    assert set(artifacts) == {"results.json", "metrics.csv"}


def test_format_metrics_uses_two_decimal_places(tmp_path: Path) -> None:
    summary = RunSummary(
        run_id="run-1",
        run_dir=tmp_path,
        status="completed",
        frame_count=30,
        average_fps=4.221,
        average_inference_ms=148.218,
        p95_total_ms=317.615,
        planning_result=PlannedPath(
            cells=[(1, 1)],
            path_length_cells=0.0,
            planning_ms=1.25,
            success=True,
        ),
        task=SemanticTask(
            target="chair",
            avoid_classes=["person"],
            speed_mode="normal",
            clarification_required=False,
        ),
    )

    assert format_metrics(summary) == {
        "平均FPS": "4.22",
        "YOLO平均推理": "148.22 ms",
        "P95总延迟": "317.62 ms",
        "处理帧数": "30",
    }


def test_load_json_reports_invalid_file_name(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("not json", encoding="utf-8")

    with pytest.raises(ValueError, match="broken.json"):
        load_json(path)


def test_load_json_requires_object_root(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text(json.dumps([1, 2]), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON对象"):
        load_json(path)
