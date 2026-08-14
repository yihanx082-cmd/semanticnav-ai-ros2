import numpy as np

from semanticnav.mapping import (
    build_semantic_grid,
    image_observation_to_cell,
)
from semanticnav.models import BBox, TrackedObject
from semanticnav.rendering import render_semantic_map


def _object(
    class_name: str,
    depth_level: str,
    center_x: float = 320,
) -> TrackedObject:
    return TrackedObject(
        track_id=1,
        class_id=0,
        class_name=class_name,
        confidence=0.9,
        bbox=BBox(
            x1=center_x - 20,
            y1=10,
            x2=center_x + 20,
            y2=50,
        ),
        depth_level=depth_level,
    )


def test_near_center_maps_near_robot() -> None:
    cell = image_observation_to_cell(320, 640, "near", (30, 30))

    assert cell == (25, 15)


def test_unknown_depth_does_not_map_to_cell() -> None:
    assert image_observation_to_cell(320, 640, "unknown", (30, 30)) is None


def test_horizontal_mapping_stays_inside_safe_columns() -> None:
    assert image_observation_to_cell(-100, 640, "mid", (30, 30)) == (16, 2)
    assert image_observation_to_cell(1000, 640, "mid", (30, 30)) == (16, 27)


def test_build_semantic_grid_assigns_class_and_avoid_costs() -> None:
    objects = [
        _object("chair", "near", center_x=160),
        _object("person", "mid", center_x=320),
        _object("couch", "far", center_x=480),
    ]

    grid = build_semantic_grid(
        objects,
        frame_width=640,
        avoid_classes={"couch"},
    )

    assert grid[25, 8] == 100
    assert grid[16, 15] == 200
    assert grid[7, 21] == 200
    assert grid.dtype == np.uint8


def test_unknown_depth_does_not_create_fake_obstacle() -> None:
    grid = build_semantic_grid(
        [_object("chair", "unknown")],
        frame_width=640,
        avoid_classes=set(),
    )

    assert np.count_nonzero(grid) == 0


def test_render_semantic_map_marks_grid_without_modifying_input() -> None:
    grid = np.zeros((30, 30), dtype=np.uint8)
    grid[25, 15] = 200
    original = grid.copy()

    image = render_semantic_map(grid)

    assert np.array_equal(grid, original)
    assert image.ndim == 3
    assert image.shape[0] > grid.shape[0]
    assert image.shape[1] > grid.shape[1]
    assert np.count_nonzero(image) > 0
