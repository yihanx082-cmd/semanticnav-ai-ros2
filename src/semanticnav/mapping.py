"""Deterministic image-observation to non-metric local grid mapping."""

import numpy as np
from numpy.typing import NDArray

from semanticnav.models import DepthLevel, TrackedObject


FREE_COST = 0
OBSTACLE_COST = 100
AVOID_COST = 200
BLOCKED_COST = 255
SEMANTIC_MAP_TITLE = "局部示意地图（非米制）"


def image_observation_to_cell(
    center_x_px: float,
    frame_width: int,
    depth_level: DepthLevel,
    grid_shape: tuple[int, int],
) -> tuple[int, int] | None:
    if depth_level == "unknown":
        return None
    if frame_width <= 0:
        raise ValueError("frame_width must be greater than 0")

    rows, columns = grid_shape
    if rows < 3 or columns < 5:
        raise ValueError("grid_shape is too small")
    row_ratios = {"near": 25 / 29, "mid": 16 / 29, "far": 7 / 29}
    row = int(np.floor((rows - 1) * row_ratios[depth_level] + 0.5))

    safe_left = 2
    safe_right = columns - 3
    normalized_x = float(np.clip(center_x_px / frame_width, 0.0, 1.0))
    column_value = safe_left + normalized_x * (safe_right - safe_left)
    column = int(np.floor(column_value + 0.5))
    return row, column


def build_semantic_grid(
    objects: list[TrackedObject],
    frame_width: int,
    avoid_classes: set[str] | list[str],
    grid_shape: tuple[int, int] = (30, 30),
) -> NDArray[np.uint8]:
    grid = np.zeros(grid_shape, dtype=np.uint8)
    avoid = {name.lower() for name in avoid_classes}
    high_risk = {"person", "cat", "dog"}

    for obj in objects:
        center_x = (obj.bbox.x1 + obj.bbox.x2) / 2.0
        cell = image_observation_to_cell(
            center_x,
            frame_width,
            obj.depth_level,
            grid_shape,
        )
        if cell is None:
            continue
        class_name = obj.class_name.lower()
        cost = (
            AVOID_COST
            if class_name in avoid or class_name in high_risk
            else OBSTACLE_COST
        )
        row, column = cell
        grid[row, column] = max(int(grid[row, column]), cost)
    return grid
