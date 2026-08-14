"""Validated video input and output helpers."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


VideoFrame = NDArray[np.uint8]


@dataclass(frozen=True)
class VideoMetadata:
    """Video properties shared by readers, processors, and writers."""

    width: int
    height: int
    fps: float
    frame_count: int

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("视频宽度必须大于 0")
        if self.height <= 0:
            raise ValueError("视频高度必须大于 0")
        if self.fps <= 0:
            raise ValueError("视频 FPS 必须大于 0")
        if self.frame_count < 0:
            raise ValueError("视频帧数不能小于 0")


def _metadata_from_capture(capture: cv2.VideoCapture, path: Path) -> VideoMetadata:
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
        )
    except ValueError as error:
        raise ValueError(f"视频元数据无效: {path}: {error}") from error


def read_video(path: str | Path) -> Iterator[tuple[int, float, VideoFrame]]:
    """Yield ``(frame_index, timestamp_s, frame)`` and release the capture."""

    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"无法打开视频: {video_path}")

    try:
        metadata = _metadata_from_capture(capture, video_path)
        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            timestamp_s = frame_index / metadata.fps
            yield frame_index, timestamp_s, frame
            frame_index += 1
    finally:
        capture.release()


def open_video_writer(
    path: str | Path,
    metadata: VideoMetadata,
    codec: str = "mp4v",
) -> cv2.VideoWriter:
    """Create a validated OpenCV writer using the supplied metadata."""

    if len(codec) != 4:
        raise ValueError("视频编码标识必须恰好包含 4 个字符")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        metadata.fps,
        (metadata.width, metadata.height),
    )
    if not writer.isOpened():
        writer.release()
        raise ValueError(f"无法创建输出视频: {output_path}")
    return writer
