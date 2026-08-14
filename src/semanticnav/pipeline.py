"""End-to-end orchestration for the fast-track SemanticNav demo."""

from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field

from semanticnav.depth import assign_depth_levels
from semanticnav.language import parse_task
from semanticnav.mapping import (
    SEMANTIC_MAP_TITLE,
    build_semantic_grid,
    image_observation_to_cell,
)
from semanticnav.models import FrameResult, PlannedPath, SemanticTask, TrackedObject
from semanticnav.planning import inflate_obstacles, plan_astar
from semanticnav.rendering import draw_tracks, render_semantic_map
from semanticnav.serialization import (
    create_run_directory,
    write_json,
    write_metrics_csv,
    write_path_csv,
    write_results_json,
)
from semanticnav.video import get_video_metadata, open_video_writer, read_video


ProgressCallback = Callable[[int, int], bool | None]


class RunSummary(BaseModel):
    run_id: str
    run_dir: Path
    status: Literal["completed", "cancelled"]
    frame_count: int = Field(ge=0)
    average_fps: float = Field(ge=0.0)
    average_inference_ms: float = Field(ge=0.0)
    p95_total_ms: float = Field(ge=0.0)
    planning_result: PlannedPath
    task: SemanticTask


def _target_matches(target: str, class_name: str) -> bool:
    aliases = {
        "sofa": {"sofa", "couch"},
        "chair": {"chair"},
        "table": {"table", "dining table", "desk"},
    }
    return class_name.lower() in aliases.get(target, {target})


def _nearest_free_cell(
    grid: np.ndarray,
    desired: tuple[int, int],
    start: tuple[int, int],
) -> tuple[int, int]:
    free_cells = [
        (row, column)
        for row in range(grid.shape[0])
        for column in range(grid.shape[1])
        if int(grid[row, column]) < 200 and (row, column) != start
    ]
    if not free_cells:
        return desired
    return min(
        free_cells,
        key=lambda cell: (
            max(abs(cell[0] - desired[0]), abs(cell[1] - desired[1])),
            abs(cell[0] - start[0]) + abs(cell[1] - start[1]),
        ),
    )


def _select_goal(
    task: SemanticTask,
    objects: list[TrackedObject],
    frame_width: int,
    inflated_grid: np.ndarray,
    start: tuple[int, int],
) -> tuple[int, int]:
    desired = (0, inflated_grid.shape[1] // 2)
    if task.target is not None:
        candidates = [
            obj for obj in objects if _target_matches(task.target, obj.class_name)
        ]
        if candidates:
            target = max(candidates, key=lambda obj: obj.confidence)
            target_cell = image_observation_to_cell(
                (target.bbox.x1 + target.bbox.x2) / 2.0,
                frame_width,
                target.depth_level,
                inflated_grid.shape,
            )
            if target_cell is not None:
                desired = target_cell
    return _nearest_free_cell(inflated_grid, desired, start)


class VideoPipeline:
    def __init__(
        self,
        tracker: Any,
        depth_estimator: Any | None,
        *,
        depth_frame_interval: int = 3,
        near_threshold: float = 0.70,
        far_threshold: float = 0.35,
        grid_shape: tuple[int, int] = (30, 30),
        obstacle_inflation_cells: int = 2,
    ) -> None:
        self.tracker = tracker
        self.depth_estimator = depth_estimator
        self.depth_frame_interval = depth_frame_interval
        self.near_threshold = near_threshold
        self.far_threshold = far_threshold
        self.grid_shape = grid_shape
        self.obstacle_inflation_cells = obstacle_inflation_cells

    def run(
        self,
        input_path: str | Path,
        task_text: str,
        output_root: str | Path,
        progress: ProgressCallback | None = None,
        *,
        start_frame: int = 0,
        max_frames: int | None = None,
    ) -> RunSummary:
        run_id, run_dir = create_run_directory(output_root)
        task = parse_task(task_text)
        frames: list[FrameResult] = []
        histories: dict[int, list[tuple[int, int]]] = {}
        latest_depth = None
        latest_objects: list[TrackedObject] = []
        writer = None
        status: Literal["completed", "cancelled"] = "completed"
        processing_started = perf_counter()

        try:
            metadata = get_video_metadata(input_path)
            writer = open_video_writer(run_dir / "annotated.mp4", metadata)
            available_frames = max(0, metadata.frame_count - start_frame)
            total_frames = (
                min(available_frames, max_frames)
                if max_frames is not None
                else available_frames
            )

            for frame_index, timestamp_s, frame in read_video(input_path):
                if frame_index < start_frame:
                    continue
                frame_started = perf_counter()
                objects, inference_ms = self.tracker.track(frame)
                processed_index = len(frames)
                if self.depth_estimator is not None and (
                    latest_depth is None
                    or processed_index % self.depth_frame_interval == 0
                ):
                    latest_depth = self.depth_estimator.infer(frame)
                if latest_depth is not None:
                    objects = assign_depth_levels(
                        objects,
                        latest_depth,
                        self.near_threshold,
                        self.far_threshold,
                    )

                rendered = draw_tracks(frame, objects, histories)
                writer.write(rendered)
                total_ms = (perf_counter() - frame_started) * 1000.0
                frames.append(
                    FrameResult(
                        frame_index=frame_index,
                        timestamp_s=timestamp_s,
                        inference_ms=inference_ms,
                        total_ms=total_ms,
                        objects=objects,
                    )
                )
                latest_objects = objects

                if progress is not None and progress(len(frames), total_frames) is False:
                    status = "cancelled"
                    break
                if max_frames is not None and len(frames) >= max_frames:
                    break

            processing_elapsed = perf_counter() - processing_started
            semantic_grid = build_semantic_grid(
                latest_objects,
                metadata.width,
                task.avoid_classes,
                self.grid_shape,
            )
            inflated_grid = inflate_obstacles(
                semantic_grid,
                self.obstacle_inflation_cells,
            )
            start = (self.grid_shape[0] - 1, self.grid_shape[1] // 2)
            goal = _select_goal(
                task,
                latest_objects,
                metadata.width,
                inflated_grid,
                start,
            )
            planning_result = plan_astar(inflated_grid, start, goal)
            map_image = render_semantic_map(
                inflated_grid,
                path_cells=planning_result.cells,
            )
            if not cv2.imwrite(str(run_dir / "semantic_map.png"), map_image):
                raise ValueError("无法写出语义地图")

            average_fps = len(frames) / processing_elapsed if processing_elapsed else 0.0
            inference_values = [frame.inference_ms for frame in frames]
            total_values = [frame.total_ms for frame in frames]
            summary = RunSummary(
                run_id=run_id,
                run_dir=run_dir,
                status=status,
                frame_count=len(frames),
                average_fps=average_fps,
                average_inference_ms=(
                    float(np.mean(inference_values)) if inference_values else 0.0
                ),
                p95_total_ms=(
                    float(np.percentile(total_values, 95)) if total_values else 0.0
                ),
                planning_result=planning_result,
                task=task,
            )
            result_metadata = {
                "run_id": run_id,
                "status": status,
                "input_path": str(input_path),
                "task": task.model_dump(mode="json"),
                "depth_mode": (
                    "relative" if self.depth_estimator is not None else "disabled"
                ),
                "semantic_map_title": SEMANTIC_MAP_TITLE,
            }
            write_results_json(run_dir / "results.json", result_metadata, frames)
            write_metrics_csv(run_dir / "metrics.csv", frames)
            write_path_csv(run_dir / "path.csv", planning_result)
            write_json(
                run_dir / "run_metadata.json",
                summary.model_dump(mode="json"),
            )
            return summary
        except Exception as error:
            write_json(
                run_dir / "run_metadata.json",
                {
                    "run_id": run_id,
                    "status": "failed",
                    "frame_count": len(frames),
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise
        finally:
            if writer is not None:
                writer.release()
            self.tracker.reset()
