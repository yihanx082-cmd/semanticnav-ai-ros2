"""Relative monocular depth estimation and object risk assignment."""

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from semanticnav.models import BBox, TrackedObject


class DepthModelUnavailable(RuntimeError):
    """Raised when the optional monocular depth model cannot be loaded."""


def normalize_relative_depth(raw_depth: NDArray[np.floating]) -> NDArray[np.float32]:
    raw = np.asarray(raw_depth, dtype=np.float32)
    result = np.zeros(raw.shape, dtype=np.float32)
    finite = np.isfinite(raw)
    if not finite.any():
        return result

    minimum = float(raw[finite].min())
    maximum = float(raw[finite].max())
    span = maximum - minimum
    if span <= np.finfo(np.float32).eps:
        return result
    result[finite] = (raw[finite] - minimum) / span
    return result


def robust_object_depth(
    depth_map: NDArray[np.floating],
    bbox: BBox,
    center_fraction: float = 0.4,
    min_valid_ratio: float = 0.2,
) -> float | None:
    if not 0.0 < center_fraction <= 1.0:
        raise ValueError("center_fraction must be in (0, 1]")
    if not 0.0 <= min_valid_ratio <= 1.0:
        raise ValueError("min_valid_ratio must be between 0 and 1")

    height, width = depth_map.shape[:2]
    center_x = (bbox.x1 + bbox.x2) / 2.0
    center_y = (bbox.y1 + bbox.y2) / 2.0
    half_width = (bbox.x2 - bbox.x1) * center_fraction / 2.0
    half_height = (bbox.y2 - bbox.y1) * center_fraction / 2.0
    x1 = max(0, int(np.floor(center_x - half_width)))
    y1 = max(0, int(np.floor(center_y - half_height)))
    x2 = min(width, int(np.ceil(center_x + half_width)))
    y2 = min(height, int(np.ceil(center_y + half_height)))
    if x2 <= x1 or y2 <= y1:
        return None

    region = np.asarray(depth_map[y1:y2, x1:x2], dtype=np.float32)
    finite = np.isfinite(region)
    if finite.size == 0 or finite.mean() < min_valid_ratio:
        return None
    return float(np.median(region[finite]))


def assign_depth_levels(
    objects: list[TrackedObject],
    depth_map: NDArray[np.floating],
    near_threshold: float,
    far_threshold: float,
) -> list[TrackedObject]:
    if far_threshold >= near_threshold:
        raise ValueError("far_threshold must be smaller than near_threshold")

    assigned: list[TrackedObject] = []
    for obj in objects:
        value = robust_object_depth(depth_map, obj.bbox)
        if value is None:
            assigned.append(
                obj.model_copy(
                    update={"relative_depth": None, "depth_level": "unknown"}
                )
            )
        elif value >= near_threshold:
            assigned.append(
                obj.model_copy(update={"relative_depth": value, "depth_level": "near"})
            )
        elif value <= far_threshold:
            assigned.append(
                obj.model_copy(update={"relative_depth": value, "depth_level": "far"})
            )
        else:
            assigned.append(
                obj.model_copy(update={"relative_depth": value, "depth_level": "mid"})
            )
    return assigned


class RelativeDepthEstimator:
    """Depth Anything V2 Small wrapper that returns relative depth in [0, 1]."""

    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
    ) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            self._torch = torch
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
            self.model.eval()
        except Exception as error:
            raise DepthModelUnavailable(
                f"无法加载相对深度模型 {model_name}: {error}"
            ) from error

    def infer(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs: dict[str, Any] = self.processor(
            images=rgb_frame,
            return_tensors="pt",
        )
        with self._torch.inference_mode():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
            resized = self._torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
        raw_depth = resized.squeeze().cpu().numpy()
        return normalize_relative_depth(raw_depth)
