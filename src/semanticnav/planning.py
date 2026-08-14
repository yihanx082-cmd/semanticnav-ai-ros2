"""Obstacle inflation and safe eight-connected A* planning."""

import heapq
import math
from time import perf_counter

import cv2
import numpy as np
from numpy.typing import NDArray

from semanticnav.models import PlannedPath


GridCell = tuple[int, int]


def inflate_obstacles(
    grid: NDArray[np.uint8],
    radius_cells: int,
) -> NDArray[np.uint8]:
    if radius_cells < 0:
        raise ValueError("radius_cells cannot be negative")
    inflated = grid.copy()
    if radius_cells == 0:
        return inflated

    obstacle_mask = (grid >= 100).astype(np.uint8)
    kernel_size = radius_cells * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    expanded = cv2.dilate(obstacle_mask, kernel, iterations=1).astype(bool)
    inflated[expanded] = 255
    return inflated


def _octile_distance(first: GridCell, second: GridCell) -> float:
    row_distance = abs(first[0] - second[0])
    column_distance = abs(first[1] - second[1])
    diagonal = min(row_distance, column_distance)
    straight = max(row_distance, column_distance) - diagonal
    return diagonal * math.sqrt(2) + straight


def _failure(reason: str, started: float) -> PlannedPath:
    return PlannedPath(
        cells=[],
        path_length_cells=0.0,
        planning_ms=(perf_counter() - started) * 1000.0,
        success=False,
        failure_reason=reason,
    )


def plan_astar(
    grid: NDArray[np.uint8],
    start: GridCell,
    goal: GridCell,
    blocked_cost: int = 200,
) -> PlannedPath:
    started = perf_counter()
    rows, columns = grid.shape

    def inside(cell: GridCell) -> bool:
        return 0 <= cell[0] < rows and 0 <= cell[1] < columns

    def blocked(cell: GridCell) -> bool:
        return int(grid[cell]) >= blocked_cost

    if not inside(start):
        return _failure("invalid_start", started)
    if not inside(goal):
        return _failure("invalid_goal", started)
    if blocked(start):
        return _failure("start_blocked", started)
    if blocked(goal):
        return _failure("goal_blocked", started)

    frontier: list[tuple[float, float, GridCell]] = [
        (_octile_distance(start, goal), 0.0, start)
    ]
    costs: dict[GridCell, float] = {start: 0.0}
    previous: dict[GridCell, GridCell] = {}
    movements = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),
        (1, 1, math.sqrt(2)),
    ]

    while frontier:
        _, current_cost, current = heapq.heappop(frontier)
        if current_cost > costs.get(current, math.inf):
            continue
        if current == goal:
            cells = [goal]
            while cells[-1] != start:
                cells.append(previous[cells[-1]])
            cells.reverse()
            path_length = sum(
                _octile_distance(cells[index - 1], cells[index])
                for index in range(1, len(cells))
            )
            return PlannedPath(
                cells=cells,
                path_length_cells=path_length,
                planning_ms=(perf_counter() - started) * 1000.0,
                success=True,
                failure_reason=None,
            )

        for row_step, column_step, movement_cost in movements:
            neighbor = (current[0] + row_step, current[1] + column_step)
            if not inside(neighbor) or blocked(neighbor):
                continue
            if row_step != 0 and column_step != 0:
                side_a = (current[0] + row_step, current[1])
                side_b = (current[0], current[1] + column_step)
                if blocked(side_a) or blocked(side_b):
                    continue

            semantic_penalty = int(grid[neighbor]) / 100.0
            new_cost = current_cost + movement_cost + semantic_penalty
            if new_cost >= costs.get(neighbor, math.inf):
                continue
            costs[neighbor] = new_cost
            previous[neighbor] = current
            priority = new_cost + _octile_distance(neighbor, goal)
            heapq.heappush(frontier, (priority, new_cost, neighbor))

    return _failure("no_path", started)
