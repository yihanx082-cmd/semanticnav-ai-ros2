"""Run YOLO and ByteTrack on a local video and print tracking metrics."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from semanticnav.config import load_config
from semanticnav.depth import (
    DepthModelUnavailable,
    RelativeDepthEstimator,
    assign_depth_levels,
)
from semanticnav.tracking import YOLOByteTracker
from semanticnav.video import read_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input MP4 path")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Application YAML config",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many frames",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Skip inference until this zero-based frame index",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override model.image_size for an experiment",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Override model.confidence for an experiment",
    )
    parser.add_argument(
        "--depth",
        action="store_true",
        help="Enable relative Depth Anything V2 inference",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames 必须大于 0")
    if args.start_frame < 0:
        raise SystemExit("--start-frame 不能小于 0")

    config = load_config(args.config)
    image_size = args.image_size or config.model.image_size
    confidence = (
        config.model.confidence if args.confidence is None else args.confidence
    )
    tracker = YOLOByteTracker(
        model_name=config.model.name,
        confidence=confidence,
        image_size=image_size,
        tracker_name=config.tracker.name,
    )
    depth_estimator = None
    if args.depth and config.depth.enabled:
        try:
            depth_estimator = RelativeDepthEstimator(config.depth.model_name)
        except DepthModelUnavailable as error:
            print(f"warning: {error}; depth disabled")

    print(
        f"model={config.model.name} tracker={config.tracker.name} "
        f"image_size={image_size} confidence={confidence:g}"
    )
    started = perf_counter()
    processed_frames = 0
    inference_times: list[float] = []
    observed_track_ids: set[int] = set()
    current_depth_map = None
    depth_inferences = 0

    try:
        for frame_index, timestamp_s, frame in read_video(args.input):
            if frame_index < args.start_frame:
                continue
            objects, inference_ms = tracker.track(frame)
            processed_frames += 1
            inference_times.append(inference_ms)
            if depth_estimator is not None and (
                current_depth_map is None
                or (processed_frames - 1) % config.depth.frame_interval == 0
            ):
                current_depth_map = depth_estimator.infer(frame)
                depth_inferences += 1
            if current_depth_map is not None:
                objects = assign_depth_levels(
                    objects,
                    current_depth_map,
                    near_threshold=config.depth.near_threshold,
                    far_threshold=config.depth.far_threshold,
                )
            observed_track_ids.update(obj.track_id for obj in objects)
            print(
                f"frame={frame_index} timestamp_s={timestamp_s:.3f} "
                f"objects={len(objects)} inference_ms={inference_ms:.2f}"
            )
            for obj in objects:
                bbox = obj.bbox
                print(
                    f"  id={obj.track_id} class={obj.class_name} "
                    f"confidence={obj.confidence:.3f} "
                    f"depth={obj.depth_level} relative_depth={obj.relative_depth} "
                    f"bbox=[{bbox.x1:.1f},{bbox.y1:.1f},{bbox.x2:.1f},{bbox.y2:.1f}]"
                )

            if args.max_frames is not None and processed_frames >= args.max_frames:
                break
    finally:
        tracker.reset()

    elapsed_s = perf_counter() - started
    average_fps = processed_frames / elapsed_s if elapsed_s > 0 else 0.0
    average_inference_ms = (
        float(np.mean(inference_times)) if inference_times else 0.0
    )
    print(
        f"processed_frames={processed_frames} average_fps={average_fps:.2f} "
        f"average_inference_ms={average_inference_ms:.2f} "
        f"unique_track_ids={len(observed_track_ids)} "
        f"depth_inferences={depth_inferences}"
    )


if __name__ == "__main__":
    main()
