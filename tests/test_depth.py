import numpy as np
import pytest

from semanticnav.depth import (
    assign_depth_levels,
    normalize_relative_depth,
    robust_object_depth,
)
from semanticnav.models import BBox, TrackedObject


def _object(bbox: BBox | None = None) -> TrackedObject:
    return TrackedObject(
        track_id=1,
        class_id=56,
        class_name="chair",
        confidence=0.9,
        bbox=bbox or BBox(x1=0, y1=0, x2=10, y2=10),
    )


def test_normalize_constant_map_is_finite_zero() -> None:
    raw = np.ones((4, 4), dtype=np.float32) * 3

    result = normalize_relative_depth(raw)

    assert np.all(result == 0)
    assert np.isfinite(result).all()


def test_normalize_ignores_non_finite_values() -> None:
    raw = np.array([[1.0, 2.0], [np.nan, np.inf]], dtype=np.float32)

    result = normalize_relative_depth(raw)

    assert result[0, 0] == pytest.approx(0.0)
    assert result[0, 1] == pytest.approx(1.0)
    assert result[1, 0] == pytest.approx(0.0)
    assert result[1, 1] == pytest.approx(0.0)
    assert np.isfinite(result).all()


def test_robust_object_depth_uses_finite_center_median() -> None:
    depth = np.zeros((10, 10), dtype=np.float32)
    depth[3:7, 3:7] = 0.8
    depth[3, 3] = np.nan
    depth[3, 4] = np.inf

    value = robust_object_depth(depth, _object().bbox)

    assert value == pytest.approx(0.8)


def test_robust_object_depth_returns_none_with_too_few_valid_pixels() -> None:
    depth = np.full((10, 10), np.nan, dtype=np.float32)
    depth[4, 4] = 0.8

    assert robust_object_depth(depth, _object().bbox) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.8, "near"), (0.5, "mid"), (0.2, "far")],
)
def test_assign_depth_levels_uses_configured_thresholds(
    value: float,
    expected: str,
) -> None:
    depth = np.full((10, 10), value, dtype=np.float32)

    result = assign_depth_levels(
        [_object()],
        depth,
        near_threshold=0.7,
        far_threshold=0.35,
    )

    assert result[0].depth_level == expected
    assert result[0].relative_depth == pytest.approx(value)


def test_assign_depth_levels_marks_missing_depth_unknown() -> None:
    depth = np.full((10, 10), np.nan, dtype=np.float32)

    result = assign_depth_levels(
        [_object()],
        depth,
        near_threshold=0.7,
        far_threshold=0.35,
    )

    assert result[0].depth_level == "unknown"
    assert result[0].relative_depth is None
