from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    image_size: int = Field(gt=0)


class TrackerConfig(BaseModel):
    name: str


class DepthConfig(BaseModel):
    enabled: bool
    frame_interval: int = Field(gt=0)
    near_threshold: float = Field(ge=0.0, le=1.0)
    far_threshold: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> "DepthConfig":
        if self.far_threshold >= self.near_threshold:
            raise ValueError("far_threshold must be smaller than near_threshold")
        return self


class MappingConfig(BaseModel):
    rows: int = Field(gt=0)
    columns: int = Field(gt=0)
    obstacle_inflation_cells: int = Field(ge=0)


class OutputConfig(BaseModel):
    root: Path


class AppConfig(BaseModel):
    model: ModelConfig
    tracker: TrackerConfig
    depth: DepthConfig
    mapping: MappingConfig
    output: OutputConfig


def load_config(path: Path) -> AppConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    return AppConfig.model_validate(raw_config)
