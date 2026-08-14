"""Generate a small deterministic MP4 for video I/O checks."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from semanticnav.video import VideoMetadata, open_video_writer


def generate_test_video(output_path: Path) -> VideoMetadata:
    metadata = VideoMetadata(width=320, height=240, fps=10.0, frame_count=10)
    writer = open_video_writer(output_path, metadata)
    try:
        for frame_index in range(metadata.frame_count):
            frame = np.zeros((metadata.height, metadata.width, 3), dtype=np.uint8)
            x1 = 20 + frame_index * 20
            cv2.rectangle(frame, (x1, 85), (x1 + 60, 145), (0, 255, 0), -1)
            cv2.putText(
                frame,
                f"Frame {frame_index}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/test_video.mp4"),
        help="Generated MP4 path (default: outputs/test_video.mp4)",
    )
    args = parser.parse_args()

    metadata = generate_test_video(args.output)
    print(
        f"已生成 {args.output}: {metadata.width}x{metadata.height}, "
        f"{metadata.fps:g} FPS, {metadata.frame_count} 帧"
    )


if __name__ == "__main__":
    main()
