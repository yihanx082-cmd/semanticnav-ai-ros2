import json
from pathlib import Path

import numpy as np
import pytest

from scripts.make_test_video import generate_test_video
from semanticnav.models import BBox, TrackedObject
from semanticnav.pipeline import VideoPipeline
from semanticnav.video import read_video


class FakeTracker:
    def __init__(self, fail_at: int | None = None) -> None:
        self.calls = 0
        self.fail_at = fail_at
        self.reset_count = 0

    def track(self, frame: np.ndarray) -> tuple[list[TrackedObject], float]:
        if self.fail_at is not None and self.calls == self.fail_at:
            raise RuntimeError("synthetic tracker failure")
        self.calls += 1
        return (
            [
                TrackedObject(
                    track_id=1,
                    class_id=56,
                    class_name="chair",
                    confidence=0.9,
                    bbox=BBox(x1=130, y1=80, x2=190, y2=180),
                )
            ],
            10.0,
        )

    def reset(self) -> None:
        self.reset_count += 1


class FakeDepth:
    def __init__(self) -> None:
        self.calls = 0

    def infer(self, frame: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.full(frame.shape[:2], 0.8, dtype=np.float32)


@pytest.fixture
def test_video(tmp_path: Path) -> Path:
    path = tmp_path / "input.mp4"
    generate_test_video(path)
    return path


def test_pipeline_writes_artifacts(test_video: Path, tmp_path: Path) -> None:
    tracker = FakeTracker()
    depth = FakeDepth()

    summary = VideoPipeline(tracker, depth).run(
        test_video,
        "去椅子附近，避开人",
        tmp_path / "outputs",
    )

    assert summary.status == "completed"
    assert summary.frame_count == 10
    for name in [
        "annotated.mp4",
        "results.json",
        "metrics.csv",
        "semantic_map.png",
        "path.csv",
        "run_metadata.json",
    ]:
        assert (summary.run_dir / name).exists()
    assert len(list(read_video(summary.run_dir / "annotated.mp4"))) == 10
    results = json.loads(
        (summary.run_dir / "results.json").read_text(encoding="utf-8")
    )
    assert results["frames"][0]["objects"][0]["depth_level"] == "near"
    assert "distance_m" not in results["frames"][0]["objects"][0]
    assert depth.calls == 4
    assert tracker.reset_count == 1


def test_pipeline_cancellation_releases_writer(
    test_video: Path,
    tmp_path: Path,
) -> None:
    tracker = FakeTracker()

    summary = VideoPipeline(tracker, None).run(
        test_video,
        "去椅子附近",
        tmp_path / "outputs",
        progress=lambda processed, total: processed < 3,
    )

    assert summary.status == "cancelled"
    assert summary.frame_count == 3
    assert len(list(read_video(summary.run_dir / "annotated.mp4"))) == 3
    assert tracker.reset_count == 1


def test_pipeline_failure_records_status_and_releases_writer(
    test_video: Path,
    tmp_path: Path,
) -> None:
    tracker = FakeTracker(fail_at=1)
    output_root = tmp_path / "outputs"

    with pytest.raises(RuntimeError, match="synthetic tracker failure"):
        VideoPipeline(tracker, None).run(
            test_video,
            "去椅子附近",
            output_root,
        )

    run_dirs = list(output_root.iterdir())
    metadata = json.loads(
        (run_dirs[0] / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["status"] == "failed"
    assert tracker.reset_count == 1
    assert len(list(read_video(run_dirs[0] / "annotated.mp4"))) == 1
