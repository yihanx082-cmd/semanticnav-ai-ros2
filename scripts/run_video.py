"""Run the end-to-end SemanticNav pipeline on a local video."""

import argparse
from pathlib import Path
from semanticnav.config import load_config
from semanticnav.depth import (
    DepthModelUnavailable,
    RelativeDepthEstimator,
)
from semanticnav.pipeline import VideoPipeline
from semanticnav.tracking import YOLOByteTracker


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
    parser.add_argument(
        "--task",
        default="去椅子附近，避开人和宠物",
        help="Natural-language semantic navigation task",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Root directory for run artifacts",
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
    pipeline = VideoPipeline(
        tracker,
        depth_estimator,
        depth_frame_interval=config.depth.frame_interval,
        near_threshold=config.depth.near_threshold,
        far_threshold=config.depth.far_threshold,
        grid_shape=(config.mapping.rows, config.mapping.columns),
        obstacle_inflation_cells=config.mapping.obstacle_inflation_cells,
    )

    def report_progress(processed: int, total: int) -> None:
        if processed == 1 or processed % 10 == 0 or processed == total:
            print(f"progress={processed}/{total}")

    summary = pipeline.run(
        args.input,
        args.task,
        args.output_root,
        progress=report_progress,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
    )
    print(
        f"status={summary.status} frames={summary.frame_count} "
        f"average_fps={summary.average_fps:.2f} "
        f"average_inference_ms={summary.average_inference_ms:.2f} "
        f"p95_total_ms={summary.p95_total_ms:.2f} "
        f"path_success={summary.planning_result.success}"
    )
    print(f"run_dir={summary.run_dir}")


if __name__ == "__main__":
    main()
