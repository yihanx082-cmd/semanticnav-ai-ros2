from pathlib import Path

import pytest
from pydantic import ValidationError

from semanticnav.config import load_config
from semanticnav.models import BBox, TrackedObject


def test_tracked_object_rejects_invalid_confidence():
    with pytest.raises(ValidationError):
        TrackedObject(
            track_id=1,
            class_id=0,
            class_name="person",
            confidence=1.2,
            bbox=BBox(x1=0, y1=0, x2=10, y2=10),
        )


def test_bbox_rejects_inverted_coordinates():
    with pytest.raises(ValidationError):
        BBox(x1=10, y1=0, x2=5, y2=10)


def test_default_config_uses_relative_output_path():
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "default.yaml")

    assert config.model.confidence == 0.25
    assert config.depth.frame_interval == 3
    assert config.output.root == Path("outputs")
    assert not config.output.root.is_absolute()
