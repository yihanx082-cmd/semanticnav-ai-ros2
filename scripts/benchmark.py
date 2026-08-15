"""Run reproducible SemanticNav performance scenarios on one video clip."""

import argparse
import csv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from semanticnav.config import AppConfig, load_config
from semanticnav.depth import RelativeDepthEstimator
from semanticnav.pipeline import VideoPipeline
from semanticnav.tracking import YOLOByteTracker


@dataclass(frozen=True)
class BenchmarkScenario:
    name: str
    image_size: int
    confidence: float
    depth_enabled: bool
    depth_frame_interval: int
    max_frames: int


FIELDNAMES = (
    "scenario",
    "model",
    "image_size",
    "confidence",
    "depth_enabled",
    "depth_frame_interval",
    "run_id",
    "run_dir",
    "status",
    "frame_count",
    "average_fps",
    "average_inference_ms",
    "p95_total_ms",
    "path_success",
)


def summarize_run(metadata: dict[str, object]) -> dict[str, object]:
    required = (
        "run_id",
        "status",
        "frame_count",
        "average_fps",
        "average_inference_ms",
        "p95_total_ms",
        "planning_result",
    )
    missing = [name for name in required if name not in metadata]
    if missing:
        raise ValueError(f"运行元数据缺少字段: {', '.join(missing)}")
    planning_result = metadata["planning_result"]
    if not isinstance(planning_result, Mapping) or "success" not in planning_result:
        raise ValueError("运行元数据缺少字段: planning_result.success")
    return {
        "run_id": metadata["run_id"],
        "status": metadata["status"],
        "frame_count": metadata["frame_count"],
        "average_fps": metadata["average_fps"],
        "average_inference_ms": metadata["average_inference_ms"],
        "p95_total_ms": metadata["p95_total_ms"],
        "path_success": bool(planning_result["success"]),
    }


def build_scenarios(max_frames: int) -> list[BenchmarkScenario]:
    if max_frames <= 0:
        raise ValueError("max_frames必须大于0")
    return [
        BenchmarkScenario("E1", 480, 0.25, False, 5, max_frames),
        BenchmarkScenario("E2", 640, 0.25, False, 5, max_frames),
        BenchmarkScenario("E3", 480, 0.50, False, 5, max_frames),
        BenchmarkScenario("E4", 480, 0.25, True, 5, min(max_frames, 20)),
    ]


def write_benchmark_csv(
    output_path: Path,
    rows: list[dict[str, object]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def run_scenario(
    input_path: Path,
    start_frame: int,
    scenario: BenchmarkScenario,
    config: AppConfig,
    output_root: Path,
) -> dict[str, object]:
    tracker = YOLOByteTracker(
        model_name=config.model.name,
        confidence=scenario.confidence,
        image_size=scenario.image_size,
        tracker_name=config.tracker.name,
    )
    depth_estimator = (
        RelativeDepthEstimator(config.depth.model_name)
        if scenario.depth_enabled
        else None
    )
    pipeline = VideoPipeline(
        tracker,
        depth_estimator,
        depth_frame_interval=scenario.depth_frame_interval,
        near_threshold=config.depth.near_threshold,
        far_threshold=config.depth.far_threshold,
        grid_shape=(config.mapping.rows, config.mapping.columns),
        obstacle_inflation_cells=config.mapping.obstacle_inflation_cells,
    )
    summary = pipeline.run(
        input_path,
        "去椅子附近，避开人和宠物",
        output_root,
        start_frame=start_frame,
        max_frames=scenario.max_frames,
    )
    row = {
        "scenario": scenario.name,
        "model": config.model.name,
        "image_size": scenario.image_size,
        "confidence": scenario.confidence,
        "depth_enabled": scenario.depth_enabled,
        "depth_frame_interval": scenario.depth_frame_interval,
        "run_dir": str(summary.run_dir),
    }
    row.update(summarize_run(summary.model_dump(mode="json")))
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/latest_results.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/benchmarks"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.start_frame < 0:
        raise SystemExit("--start-frame不能小于0")
    config = load_config(args.config)
    rows = []
    for scenario in build_scenarios(args.max_frames):
        print(
            f"running={scenario.name} imgsz={scenario.image_size} "
            f"conf={scenario.confidence:g} depth={scenario.depth_enabled} "
            f"frames={scenario.max_frames}"
        )
        row = run_scenario(
            args.input,
            args.start_frame,
            scenario,
            config,
            args.output_root,
        )
        rows.append(row)
        print(
            f"completed={scenario.name} fps={float(row['average_fps']):.3f} "
            f"p95_ms={float(row['p95_total_ms']):.3f}"
        )
    write_benchmark_csv(args.output, rows)
    print(f"benchmark_csv={args.output}")


if __name__ == "__main__":
    main()
