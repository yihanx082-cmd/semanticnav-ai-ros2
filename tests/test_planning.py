import numpy as np
import pytest

from semanticnav.planning import inflate_obstacles, plan_astar


def test_astar_finds_path_in_empty_grid() -> None:
    grid = np.zeros((5, 5), dtype=np.uint8)

    result = plan_astar(grid, (4, 2), (0, 2))

    assert result.success is True
    assert result.cells[0] == (4, 2)
    assert result.cells[-1] == (0, 2)
    assert result.path_length_cells == pytest.approx(4.0)


def test_astar_returns_no_path_for_closed_map() -> None:
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[2, :] = 255

    result = plan_astar(grid, (4, 2), (0, 2))

    assert result.success is False
    assert result.failure_reason == "no_path"
    assert result.cells == []


def test_astar_does_not_cut_blocked_corner() -> None:
    grid = np.zeros((3, 3), dtype=np.uint8)
    grid[1, 0] = 255
    grid[2, 1] = 255

    result = plan_astar(grid, (2, 0), (0, 2))

    assert result.success is False
    assert result.failure_reason == "no_path"


def test_inflate_obstacles_expands_without_modifying_input() -> None:
    grid = np.zeros((7, 7), dtype=np.uint8)
    grid[3, 3] = 100
    original = grid.copy()

    inflated = inflate_obstacles(grid, radius_cells=1)

    assert np.array_equal(grid, original)
    assert np.all(inflated[2:5, 2:5] == 255)
    assert np.count_nonzero(inflated) == 9


@pytest.mark.parametrize(
    ("start", "goal", "failure_reason"),
    [
        ((-1, 0), (0, 0), "invalid_start"),
        ((0, 0), (5, 0), "invalid_goal"),
    ],
)
def test_astar_reports_invalid_endpoint(
    start: tuple[int, int],
    goal: tuple[int, int],
    failure_reason: str,
) -> None:
    grid = np.zeros((5, 5), dtype=np.uint8)

    result = plan_astar(grid, start, goal)

    assert result.success is False
    assert result.failure_reason == failure_reason


def test_astar_reports_blocked_start_separately() -> None:
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[4, 2] = 255

    result = plan_astar(grid, (4, 2), (0, 2))

    assert result.success is False
    assert result.failure_reason == "start_blocked"
