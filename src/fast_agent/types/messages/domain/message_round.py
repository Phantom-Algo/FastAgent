from typing import List

from pydantic import BaseModel

from ..base_message import BaseMessage
from .assistant_message import AssistantMessage
from .tool_result_message import ToolResultMessage
from .user_message import UserMessage


class MessageRound(BaseModel):
    """封装一个完整的消息轮次。"""

    round_index: int
    start_message_index: int
    end_message_index: int
    messages: List[BaseMessage]
    user_message: UserMessage
    assistant_messages: List[AssistantMessage]
    tool_result_messages: List[ToolResultMessage]

    @property
    def has_tool_calls(self) -> bool:
        return any(message.tool_calls for message in self.assistant_messages)
