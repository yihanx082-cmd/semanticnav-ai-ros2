from typing import Literal

from pydantic import BaseModel, Field, model_validator


DepthLevel = Literal["near", "mid", "far", "unknown"]


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def validate_coordinates(self) -> "BBox":
        if self.x2 <= self.x1:
            raise ValueError("x2 must be greater than x1")
        if self.y2 <= self.y1:
            raise ValueError("y2 must be greater than y1")
        return self


class TrackedObject(BaseModel):
    track_id: int
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BBox
    depth_level: DepthLevel = "unknown"
    relative_depth: float | None = Field(default=None, ge=0.0, le=1.0)


class FrameResult(BaseModel):
    frame_index: int = Field(ge=0)
    timestamp_s: float = Field(ge=0.0)
    inference_ms: float = Field(ge=0.0)
    total_ms: float = Field(ge=0.0)
    objects: list[TrackedObject]


class SemanticTask(BaseModel):
    target: str | None
    avoid_classes: list[str]
    speed_mode: Literal["slow", "normal"]
    clarification_required: bool
    clarification_reason: str | None = None


class PlannedPath(BaseModel):
    cells: list[tuple[int, int]]
    path_length_cells: float = Field(ge=0.0)
    planning_ms: float = Field(ge=0.0)
    success: bool
    failure_reason: str | None = None
