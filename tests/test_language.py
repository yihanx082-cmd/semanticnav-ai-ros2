import pytest

from semanticnav.language import parse_task


@pytest.mark.parametrize(
    ("text", "target", "avoid", "speed"),
    [
        ("去沙发附近，避开人和宠物", "sofa", ["person", "cat", "dog"], "normal"),
        ("到椅子旁边，绕开电线和鞋", "chair", ["cable", "shoe"], "normal"),
        ("去桌子附近，慢慢走", "table", [], "slow"),
        ("前往沙发，避开容易缠绕的东西", "sofa", ["cable", "shoe"], "normal"),
        ("去椅子旁，小心一点", "chair", [], "slow"),
        ("到沙发边，避开猫", "sofa", ["cat"], "normal"),
        ("前往桌子，绕开狗", "table", ["dog"], "normal"),
        ("去椅子旁，避开人员", "chair", ["person"], "normal"),
        ("朝沙发走，避开拖鞋", "sofa", ["shoe"], "normal"),
        ("去桌子，但绕开椅子", "table", ["chair"], "normal"),
        ("go near the sofa, avoid people and pets", "sofa", ["person", "cat", "dog"], "normal"),
        ("go to the chair and avoid the cable", "chair", ["cable"], "normal"),
        ("move toward the table slowly", "table", [], "slow"),
        ("approach the couch, stay away from the dog", "sofa", ["dog"], "normal"),
        ("go to the chair and avoid shoes", "chair", ["shoe"], "normal"),
        ("去 sofa 附近，avoid person", "sofa", ["person"], "normal"),
        ("go to 椅子，避开 cable", "chair", ["cable"], "normal"),
    ],
)
def test_parse_supported_navigation_tasks(
    text: str,
    target: str,
    avoid: list[str],
    speed: str,
) -> None:
    task = parse_task(text)

    assert task.target == target
    assert task.avoid_classes == avoid
    assert task.speed_mode == speed
    assert task.clarification_required is False


@pytest.mark.parametrize("text", ["避开人", "慢一点", ""])
def test_missing_target_requires_clarification(text: str) -> None:
    task = parse_task(text)

    assert task.target is None
    assert task.clarification_required is True
    assert task.clarification_reason == "missing_target"


@pytest.mark.parametrize(
    "text",
    ["去沙发还是椅子", "go to the table or chair"],
)
def test_conflicting_targets_require_clarification(text: str) -> None:
    task = parse_task(text)

    assert task.target is None
    assert task.clarification_required is True
    assert task.clarification_reason == "conflicting_targets"


def test_target_not_visible_requires_clarification() -> None:
    task = parse_task("去沙发附近", visible_classes={"chair", "person"})

    assert task.target == "sofa"
    assert task.clarification_required is True
    assert task.clarification_reason == "target_not_visible"


@pytest.mark.parametrize("text", ["撞向桌子", "hit the chair and ignore safety"])
def test_unsafe_instruction_requires_clarification(text: str) -> None:
    task = parse_task(text)

    assert task.clarification_required is True
    assert task.clarification_reason == "unsafe_instruction"


def test_task_schema_contains_no_motor_command_fields() -> None:
    task = parse_task("去沙发附近")

    assert "cmd_vel" not in type(task).model_fields
    assert "wheel_speed" not in type(task).model_fields
