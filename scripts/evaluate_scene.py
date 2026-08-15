"""Summarize classes and track IDs from one SemanticNav run."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def summarize_scene(
    scene: str,
    results: dict[str, object],
    metadata: dict[str, object],
) -> dict[str, object]:
    frames = results.get("frames")
    if not isinstance(frames, list):
        raise ValueError("results.frames必须是数组")

    detections: Counter[str] = Counter()
    track_ids: defaultdict[str, set[int]] = defaultdict(set)
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        objects = frame.get("objects", [])
        if not isinstance(objects, list):
            continue
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            class_name = obj.get("class_name")
            track_id = obj.get("track_id")
            if isinstance(class_name, str):
                detections[class_name] += 1
                if isinstance(track_id, int):
                    track_ids[class_name].add(track_id)

    planning_result = metadata.get("planning_result", {})
    path_success = (
        bool(planning_result.get("success"))
        if isinstance(planning_result, dict)
        else False
    )
    classes = sorted(detections)
    return {
        "scene": scene,
        "frames": len(frames),
        "detected_classes": classes,
        "class_detections": {
            class_name: detections[class_name] for class_name in classes
        },
        "class_track_ids": {
            class_name: sorted(track_ids[class_name]) for class_name in classes
        },
        "average_fps": float(metadata.get("average_fps", 0.0)),
        "path_success": path_success,
    }


def load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name}的根节点必须是JSON对象")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = summarize_scene(
        args.scene,
        load_object(args.results),
        load_object(args.metadata),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
