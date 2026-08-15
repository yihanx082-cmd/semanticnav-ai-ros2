from scripts.evaluate_scene import summarize_scene


def test_summarize_scene_counts_classes_and_track_ids() -> None:
    results = {
        "frames": [
            {
                "objects": [
                    {"class_name": "person", "track_id": 1},
                    {"class_name": "bottle", "track_id": 2},
                ]
            },
            {
                "objects": [
                    {"class_name": "person", "track_id": 1},
                    {"class_name": "bottle", "track_id": 3},
                ]
            },
        ]
    }
    metadata = {
        "average_fps": 1.25,
        "planning_result": {"success": True},
    }

    summary = summarize_scene("person", results, metadata)

    assert summary == {
        "scene": "person",
        "frames": 2,
        "detected_classes": ["bottle", "person"],
        "class_detections": {"bottle": 2, "person": 2},
        "class_track_ids": {"bottle": [2, 3], "person": [1]},
        "average_fps": 1.25,
        "path_success": True,
    }


def test_summarize_scene_accepts_empty_frames() -> None:
    summary = summarize_scene(
        "empty",
        {"frames": []},
        {"average_fps": 0.0, "planning_result": {"success": False}},
    )

    assert summary["frames"] == 0
    assert summary["detected_classes"] == []
    assert summary["class_track_ids"] == {}
    assert summary["path_success"] is False
