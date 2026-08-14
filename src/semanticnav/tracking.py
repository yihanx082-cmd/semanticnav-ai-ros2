"""Ultralytics YOLO and ByteTrack adapter."""

from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from semanticnav.models import BBox, TrackedObject


ClassNames = Mapping[int, str] | Sequence[str]


def _as_numpy(value: Any) -> NDArray[np.generic]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _class_name(names: ClassNames, class_id: int) -> str:
    if isinstance(names, Mapping):
        return names.get(class_id, str(class_id))
    if 0 <= class_id < len(names):
        return names[class_id]
    return str(class_id)


def convert_ultralytics_result(
    result: Any,
    names: ClassNames,
) -> list[TrackedObject]:
    """Convert one tracked Ultralytics result to project data models."""

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    track_ids = getattr(boxes, "id", None)
    if track_ids is None:
        return []

    coordinates = _as_numpy(boxes.xyxy).reshape(-1, 4)
    confidences = _as_numpy(boxes.conf).reshape(-1)
    class_ids = _as_numpy(boxes.cls).reshape(-1)
    track_ids_array = _as_numpy(track_ids).reshape(-1)
    object_count = min(
        len(coordinates),
        len(confidences),
        len(class_ids),
        len(track_ids_array),
    )

    objects: list[TrackedObject] = []
    for index in range(object_count):
        raw_track_id = float(track_ids_array[index])
        if not np.isfinite(raw_track_id):
            continue

        class_id = int(class_ids[index])
        x1, y1, x2, y2 = (float(value) for value in coordinates[index])
        objects.append(
            TrackedObject(
                track_id=int(raw_track_id),
                class_id=class_id,
                class_name=_class_name(names, class_id),
                confidence=float(confidences[index]),
                bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
            )
        )
    return objects


class YOLOByteTracker:
    """Keep one YOLO model and persistent ByteTrack state for a video."""

    def __init__(
        self,
        model_name: str,
        confidence: float,
        image_size: int,
        tracker_name: str = "bytetrack.yaml",
        model: Any | None = None,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if image_size <= 0:
            raise ValueError("image_size must be greater than 0")

        if model is None:
            from ultralytics import YOLO

            model = YOLO(model_name)

        self.model = model
        self.confidence = confidence
        self.image_size = image_size
        self.tracker_name = tracker_name

    def track(self, frame: NDArray[np.uint8]) -> tuple[list[TrackedObject], float]:
        started = perf_counter()
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_name,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        )
        elapsed_ms = (perf_counter() - started) * 1000.0
        if not results:
            return [], elapsed_ms

        result = results[0]
        names = getattr(result, "names", getattr(self.model, "names", {}))
        objects = convert_ultralytics_result(result, names)
        speed = getattr(result, "speed", None) or {}
        inference_ms = float(speed.get("inference", elapsed_ms))
        return objects, inference_ms

    def reset(self) -> None:
        """Clear persistent ByteTrack state without reloading model weights."""

        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", ()) if predictor is not None else ()
        for tracker in trackers:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()
