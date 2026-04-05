import os
import sys

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


from fast_agent.resources.messages.message_manager import MessageManager
from fast_agent.types.messages.domain.assistant_message import (
    AssistantMessage,
    AssistantMessageFinishReasonEnum,
    ToolCall,
)
from fast_agent.types.messages.domain.tool_result_message import ToolResultMessage
from fast_agent.types.messages.domain.user_message import UserMessage


def _build_simple_round(index: int):
    return [
        UserMessage(content=f"user-{index}"),
        AssistantMessage(
            content=f"assistant-{index}",
            finish_reason=AssistantMessageFinishReasonEnum.STOP,
        ),
    ]


def _build_tool_round(index: int):
    tool_call = ToolCall(
        tool_call_id=f"call_{index}",
        function_name="mock_tool",
        function_args={"index": index},
    )
    return [
        UserMessage(content=f"user-{index}"),
        AssistantMessage(
            content=f"assistant-tool-{index}",
            tool_calls=[tool_call],
            finish_reason=AssistantMessageFinishReasonEnum.TOOL_CALLS,
        ),
        ToolResultMessage(
            tool_call_id=tool_call.tool_call_id,
            name="mock_tool",
            content=f"tool-result-{index}",
        ),
        AssistantMessage(
            content=f"assistant-final-{index}",
            finish_reason=AssistantMessageFinishReasonEnum.STOP,
        ),
    ]


def test_message_manager_get_rounds_and_count():
    messages = []
    messages.extend(_build_simple_round(0))
    messages.extend(_build_tool_round(1))
    messages.append(UserMessage(content="pending-user"))

    manager = MessageManager(messages)

    assert manager.get_round_count() == 2

    rounds = manager.get_rounds()
    assert [item.round_index for item in rounds] == [0, 1]
    assert [item.user_message.content for item in rounds] == ["user-0", "user-1"]
    assert rounds[0].has_tool_calls is False
    assert rounds[1].has_tool_calls is True
    assert len(rounds[0].messages) == 2
    assert len(rounds[1].messages) == 4
    assert len(rounds[1].assistant_messages) == 2
    assert len(rounds[1].tool_result_messages) == 1

    partial_rounds = manager.get_rounds(1, 1)
    assert len(partial_rounds) == 1
    assert partial_rounds[0].user_message.content == "user-1"


def test_message_manager_remove_rounds_by_range():
    messages = []
    for index in range(5):
        messages.extend(_build_simple_round(index))

    manager = MessageManager(messages)

    removed_rounds = manager.remove_rounds(0, 2)

    assert [item.user_message.content for item in removed_rounds] == ["user-0", "user-1", "user-2"]
    assert manager.get_round_count() == 2
    assert manager.get_message_count() == 4
    assert [item.user_message.content for item in manager.get_rounds()] == ["user-3", "user-4"]


def test_message_manager_round_range_validation():
    manager = MessageManager(_build_simple_round(0))

    with pytest.raises(IndexError):
        manager.get_rounds(1, 1)

    with pytest.raises(IndexError):
        manager.remove_rounds(1, 0)
