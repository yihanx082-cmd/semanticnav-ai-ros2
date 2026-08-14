import numpy as np

from semanticnav.models import BBox, TrackedObject
from semanticnav.rendering import draw_tracks


def _tracked_chair(x1: float = 20, x2: float = 80) -> TrackedObject:
    return TrackedObject(
        track_id=7,
        class_id=56,
        class_name="chair",
        confidence=0.91,
        bbox=BBox(x1=x1, y1=30, x2=x2, y2=100),
        depth_level="near",
    )


def test_draw_tracks_returns_modified_copy_without_changing_input() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    original = frame.copy()

    rendered = draw_tracks(frame, [_tracked_chair()], {})

    assert np.array_equal(frame, original)
    assert not np.array_equal(rendered, original)
    assert rendered is not frame


def test_draw_tracks_clips_boxes_outside_frame() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    outside = _tracked_chair(x1=-50, x2=250)

    rendered = draw_tracks(frame, [outside], {})

    assert rendered.shape == frame.shape
    assert rendered.dtype == frame.dtype
    assert np.count_nonzero(rendered) > 0


def test_draw_tracks_keeps_at_most_thirty_history_points() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    histories: dict[int, list[tuple[int, int]]] = {}

    for offset in range(35):
        draw_tracks(frame, [_tracked_chair(x1=20 + offset, x2=80 + offset)], histories)

    assert len(histories[7]) == 30
