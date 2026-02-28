import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fast_agent.llm.schema.system_prompt import SystemPrompt


def build_prompt() -> SystemPrompt:
    return SystemPrompt(
        {
            "order": ["a", "b"],
            "splitter": "\n\n",
            "chips": {
                "a": {"name": "a", "content": "A"},
                "b": {"name": "b", "content": "B"},
            },
        }
    )


def test_insert_adds_chip_at_target_index() -> None:
    prompt = build_prompt()

    prompt.insert("x", "X", index=1)

    assert prompt.chips.order == ["a", "x", "b"]
    assert prompt.get("x") is not None
    assert prompt.get("x").content == "X"


def test_move_changes_order_position() -> None:
    prompt = build_prompt()
    prompt.insert("x", "X", index=2)

    prompt.move("x", 0)

    assert prompt.chips.order == ["x", "a", "b"]


def test_ignore_toggle_wakeup_and_wakeup_all() -> None:
    prompt = build_prompt()

    returned_key = prompt.ignore("a")
    assert returned_key == "a"
    assert prompt.get("a").metadata.ignore is True

    toggled_key = prompt.toggle("a")
    assert toggled_key == "a"
    assert prompt.get("a").metadata.ignore is False

    prompt.ignore("a")
    prompt.ignore("b")
    waked = prompt.wakeup_all()
    assert set(waked) == {"a", "b"}
    assert prompt.get("a").metadata.ignore is False
    assert prompt.get("b").metadata.ignore is False

    wakeup_key = prompt.wakeup("a")
    assert wakeup_key == "a"
    assert prompt.get("a").metadata.ignore is False


def test_replace_chips_replaces_all_content() -> None:
    prompt = build_prompt()

    prompt.replace_chips(
        {
            "order": ["new"],
            "splitter": "\n",
            "chips": {
                "new": {
                    "name": "new",
                    "content": "NEW",
                    "metadata": {"ignore": False},
                }
            },
        }
    )

    assert prompt.chips.order == ["new"]
    assert prompt.get("a") is None
    assert prompt.get_system_prompt() == "NEW"


def test_insert_and_move_raise_on_invalid_input() -> None:
    prompt = build_prompt()

    with pytest.raises(ValueError):
        prompt.insert("a", "dup", index=0)

    with pytest.raises(ValueError):
        prompt.insert("x", "X", index=10)

    with pytest.raises(ValueError):
        prompt.move("missing", 0)

    with pytest.raises(ValueError):
        prompt.move("a", 10)


def test_ignore_related_apis_raise_when_key_missing() -> None:
    prompt = build_prompt()

    with pytest.raises(ValueError):
        prompt.ignore("missing")

    with pytest.raises(ValueError):
        prompt.wakeup("missing")

    with pytest.raises(ValueError):
        prompt.toggle("missing")
