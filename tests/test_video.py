from pathlib import Path

import cv2
import numpy as np
import pytest

from semanticnav.video import VideoMetadata, open_video_writer, read_video


def _make_video(path: Path, frame_count: int = 10) -> None:
    metadata = VideoMetadata(width=320, height=240, fps=10.0, frame_count=frame_count)
    writer = open_video_writer(path, metadata)
    try:
        for frame_index in range(frame_count):
            frame = np.zeros((metadata.height, metadata.width, 3), dtype=np.uint8)
            cv2.rectangle(
                frame,
                (20 + frame_index * 8, 80),
                (80 + frame_index * 8, 140),
                (0, 255, 0),
                thickness=-1,
            )
            writer.write(frame)
    finally:
        writer.release()


@pytest.fixture
def test_video(tmp_path: Path) -> Path:
    path = tmp_path / "test.mp4"
    _make_video(path)
    return path


def test_read_video_returns_index_timestamp_and_frame(test_video: Path) -> None:
    frames = list(read_video(test_video))

    assert len(frames) == 10
    assert frames[0][0] == 0
    assert frames[1][1] == pytest.approx(0.1, abs=0.02)
    assert frames[0][2].shape == (240, 320, 3)


def test_read_video_timestamps_are_monotonic(test_video: Path) -> None:
    timestamps = [timestamp for _, timestamp, _ in read_video(test_video)]

    assert timestamps == sorted(timestamps)
    assert len(set(timestamps)) == len(timestamps)


def test_read_video_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="视频文件不存在"):
        list(read_video(tmp_path / "missing.mp4"))


def test_read_video_rejects_corrupted_file(tmp_path: Path) -> None:
    corrupted_video = tmp_path / "bad.mp4"
    corrupted_video.write_text("this is not a video", encoding="utf-8")

    with pytest.raises(ValueError, match="无法打开视频"):
        list(read_video(corrupted_video))


@pytest.mark.parametrize(
    ("field", "value"),
    [("width", 0), ("height", 0), ("fps", 0.0), ("frame_count", -1)],
)
def test_video_metadata_rejects_invalid_values(field: str, value: float) -> None:
    values = {"width": 320, "height": 240, "fps": 10.0, "frame_count": 10}
    values[field] = value

    with pytest.raises(ValueError):
        VideoMetadata(**values)


def test_open_video_writer_creates_readable_output(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "result.mp4"
    _make_video(output_path)

    frames = list(read_video(output_path))

    assert output_path.exists()
    assert len(frames) == 10
    assert frames[0][2].shape == (240, 320, 3)
