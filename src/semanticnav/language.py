"""Deterministic Chinese and English semantic navigation task parser."""

import re
from collections.abc import Iterable

from semanticnav.models import SemanticTask


TARGET_ALIASES = {
    "sofa": ("沙发", "sofa", "couch"),
    "chair": ("椅子", "chair"),
    "table": ("桌子", "餐桌", "table", "desk"),
}
AVOID_ALIASES = {
    "person": ("人员", "人", "person", "people"),
    "cat": ("猫", "cat"),
    "dog": ("狗", "dog"),
    "cable": ("电线", "线缆", "cable", "wire"),
    "shoe": ("拖鞋", "鞋子", "鞋", "shoe", "shoes", "slipper", "slippers"),
    "chair": ("椅子", "chair"),
    "sofa": ("沙发", "sofa", "couch"),
    "table": ("桌子", "餐桌", "table", "desk"),
}
AVOID_ORDER = ("person", "cat", "dog", "cable", "shoe", "chair", "sofa", "table")
AVOID_CUES = ("避开", "绕开", "远离", "avoid", "stay away")
SLOW_CUES = ("慢慢", "慢一点", "小心", "缓慢", "slow", "slowly", "carefully")
UNSAFE_CUES = ("撞向", "撞上", "碰撞", "hit ", "crash", "ignore safety", "忽略安全")
PET_ALIASES = ("宠物", "pet", "pets")
ENTANGLEMENT_ALIASES = ("容易缠绕的东西", "缠绕物", "things that tangle")


def _contains(text: str, alias: str) -> bool:
    if alias.isascii() and alias.replace(" ", "").isalpha():
        return re.search(rf"\b{re.escape(alias)}\b", text) is not None
    return alias in text


def _contains_any(text: str, aliases: Iterable[str]) -> bool:
    return any(_contains(text, alias) for alias in aliases)


def _first_avoid_index(text: str) -> int | None:
    indices = [text.index(cue) for cue in AVOID_CUES if cue in text]
    return min(indices) if indices else None


def parse_task(
    text: str,
    visible_classes: set[str] | list[str] | None = None,
) -> SemanticTask:
    normalized = text.strip().lower()
    avoid_index = _first_avoid_index(normalized)
    target_text = normalized if avoid_index is None else normalized[:avoid_index]
    avoid_text = "" if avoid_index is None else normalized[avoid_index:]

    target_candidates = [
        name
        for name, aliases in TARGET_ALIASES.items()
        if _contains_any(target_text, aliases)
    ]
    target = target_candidates[0] if len(target_candidates) == 1 else None

    avoid_found: set[str] = set()
    if avoid_text:
        for name, aliases in AVOID_ALIASES.items():
            if _contains_any(avoid_text, aliases):
                avoid_found.add(name)
        if _contains_any(avoid_text, PET_ALIASES):
            avoid_found.update(("cat", "dog"))
        if _contains_any(avoid_text, ENTANGLEMENT_ALIASES):
            avoid_found.update(("cable", "shoe"))
    avoid_classes = [name for name in AVOID_ORDER if name in avoid_found]
    speed_mode = "slow" if _contains_any(normalized, SLOW_CUES) else "normal"

    clarification_reason: str | None = None
    if _contains_any(normalized, UNSAFE_CUES):
        clarification_reason = "unsafe_instruction"
    elif len(target_candidates) > 1:
        clarification_reason = "conflicting_targets"
    elif target is None:
        clarification_reason = "missing_target"
    elif visible_classes is not None:
        visible = {name.lower() for name in visible_classes}
        visibility_aliases = {
            "sofa": {"sofa", "couch"},
            "chair": {"chair"},
            "table": {"table", "dining table", "desk"},
        }
        if not visibility_aliases[target].intersection(visible):
            clarification_reason = "target_not_visible"

    return SemanticTask(
        target=target,
        avoid_classes=avoid_classes,
        speed_mode=speed_mode,
        clarification_required=clarification_reason is not None,
        clarification_reason=clarification_reason,
    )
