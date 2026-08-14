"""Drawing helpers for tracked objects."""

from collections.abc import MutableMapping

import cv2
import numpy as np
from numpy.typing import NDArray

from semanticnav.models import TrackedObject


TrackHistories = MutableMapping[int, list[tuple[int, int]]]


def _track_color(track_id: int) -> tuple[int, int, int]:
    return (
        64 + (track_id * 47) % 192,
        64 + (track_id * 89) % 192,
        64 + (track_id * 131) % 192,
    )


def draw_tracks(
    frame: NDArray[np.uint8],
    objects: list[TrackedObject],
    histories: TrackHistories,
) -> NDArray[np.uint8]:
    """Draw boxes, labels, and bounded trajectories on a copy of ``frame``."""

    rendered = frame.copy()
    height, width = rendered.shape[:2]
    for obj in objects:
        color = _track_color(obj.track_id)
        x1 = int(np.clip(round(obj.bbox.x1), 0, width - 1))
        y1 = int(np.clip(round(obj.bbox.y1), 0, height - 1))
        x2 = int(np.clip(round(obj.bbox.x2), 0, width - 1))
        y2 = int(np.clip(round(obj.bbox.y2), 0, height - 1))
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        history = histories.setdefault(obj.track_id, [])
        history.append(center)
        if len(history) > 30:
            del history[:-30]

        if len(history) >= 2:
            points = np.asarray(history, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(rendered, [points], False, color, thickness=2)

        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, thickness=2)
        label = (
            f"{obj.class_name} #{obj.track_id} "
            f"{obj.confidence:.2f} {obj.depth_level}"
        )
        text_y = max(18, y1 - 7)
        cv2.putText(
            rendered,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return rendered
