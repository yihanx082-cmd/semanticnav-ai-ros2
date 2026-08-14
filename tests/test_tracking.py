from types import SimpleNamespace

import numpy as np
import pytest

from semanticnav.tracking import YOLOByteTracker, convert_ultralytics_result


def _result_with_one_chair() -> SimpleNamespace:
    boxes = SimpleNamespace(
        xyxy=np.array([[315.25, 180.5, 510.0, 620.75]], dtype=np.float32),
        conf=np.array([0.91], dtype=np.float32),
        cls=np.array([56.0], dtype=np.float32),
        id=np.array([7.0], dtype=np.float32),
    )
    return SimpleNamespace(
        boxes=boxes,
        names={56: "chair"},
        speed={"inference": 12.5},
    )


def test_convert_result_preserves_track_id_and_float_coordinates() -> None:
    result = _result_with_one_chair()

    objects = convert_ultralytics_result(result, names={56: "chair"})

    assert len(objects) == 1
    assert objects[0].track_id == 7
    assert objects[0].class_name == "chair"
    assert objects[0].bbox.x1 == pytest.approx(315.25)
    assert objects[0].bbox.x2 == pytest.approx(510.0)


@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(boxes=None),
        SimpleNamespace(
            boxes=SimpleNamespace(
                xyxy=np.empty((0, 4), dtype=np.float32),
                conf=np.empty(0, dtype=np.float32),
                cls=np.empty(0, dtype=np.float32),
                id=np.empty(0, dtype=np.float32),
            )
        ),
    ],
)
def test_convert_result_returns_empty_list_without_boxes(result: SimpleNamespace) -> None:
    assert convert_ultralytics_result(result, names={}) == []


def test_convert_result_skips_boxes_without_track_ids() -> None:
    result = _result_with_one_chair()
    result.boxes.id = None

    assert convert_ultralytics_result(result, names={56: "chair"}) == []


class FakeModel:
    def __init__(self, result: SimpleNamespace) -> None:
        self.result = result
        self.names = result.names
        self.track_calls: list[dict[str, object]] = []
        self.predictor = None

    def track(self, frame: np.ndarray, **kwargs: object) -> list[SimpleNamespace]:
        self.track_calls.append({"frame": frame, **kwargs})
        return [self.result]


def test_tracker_keeps_state_and_returns_inference_time() -> None:
    model = FakeModel(_result_with_one_chair())
    tracker = YOLOByteTracker(
        model_name="unused.pt",
        confidence=0.25,
        image_size=640,
        tracker_name="bytetrack.yaml",
        model=model,
    )
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    objects, inference_ms = tracker.track(frame)

    assert objects[0].track_id == 7
    assert inference_ms == pytest.approx(12.5)
    assert model.track_calls[0]["persist"] is True
    assert model.track_calls[0]["tracker"] == "bytetrack.yaml"
    assert model.track_calls[0]["conf"] == pytest.approx(0.25)
    assert model.track_calls[0]["imgsz"] == 640
    assert model.track_calls[0]["verbose"] is False


class FakeInternalTracker:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


def test_reset_clears_existing_ultralytics_trackers() -> None:
    model = FakeModel(_result_with_one_chair())
    internal_tracker = FakeInternalTracker()
    model.predictor = SimpleNamespace(trackers=[internal_tracker])
    tracker = YOLOByteTracker(
        model_name="unused.pt",
        confidence=0.25,
        image_size=640,
        tracker_name="bytetrack.yaml",
        model=model,
    )

    tracker.reset()

    assert internal_tracker.reset_count == 1
