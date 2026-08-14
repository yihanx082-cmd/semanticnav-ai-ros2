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


def render_semantic_map(
    grid: NDArray[np.uint8],
    cell_size: int = 16,
) -> NDArray[np.uint8]:
    """Render a labeled local semantic sketch map as a BGR image."""

    rows, columns = grid.shape
    top_margin = 52
    bottom_margin = 38
    map_height = rows * cell_size
    map_width = columns * cell_size
    image = np.full(
        (top_margin + map_height + bottom_margin, map_width, 3),
        255,
        dtype=np.uint8,
    )
    colors = {
        0: (245, 245, 245),
        100: (0, 210, 255),
        200: (0, 90, 255),
        255: (40, 40, 40),
    }
    for row in range(rows):
        for column in range(columns):
            value = int(grid[row, column])
            color = colors[255 if value >= 255 else 200 if value >= 200 else 100 if value >= 100 else 0]
            x1 = column * cell_size
            y1 = top_margin + row * cell_size
            cv2.rectangle(
                image,
                (x1, y1),
                (x1 + cell_size - 1, y1 + cell_size - 1),
                color,
                thickness=-1,
            )
            cv2.rectangle(
                image,
                (x1, y1),
                (x1 + cell_size - 1, y1 + cell_size - 1),
                (210, 210, 210),
                thickness=1,
            )

    robot_center = (
        (columns // 2) * cell_size + cell_size // 2,
        top_margin + (rows - 1) * cell_size + cell_size // 2,
    )
    cv2.circle(image, robot_center, max(4, cell_size // 3), (255, 80, 40), -1)
    cv2.putText(
        image,
        "Local Semantic Sketch Map (Non-metric)",
        (10, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    legend_y = top_margin + map_height + 25
    cv2.putText(
        image,
        "Robot=blue  Obstacle=yellow  Avoid=orange  Blocked=black",
        (8, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return image
