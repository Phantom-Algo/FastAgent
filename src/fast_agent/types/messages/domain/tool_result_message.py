from pydantic import Field
from typing import Literal, Union, List
import uuid
from ..base_message import BaseMessage
from .content_part import BasePart

class ToolResultMessage(BaseMessage):
    """ToolResultMessage 工具调用结果消息类"""
    id: str = Field(default_factory=lambda: f"tool-result_{str(uuid.uuid4().hex[:16])}")

    type: Literal['tool_result_message'] = 'tool_result_message'

    role: Literal['tool_result'] = 'tool_result'

    tool_call_id: str

    name: str

    content: Union[str, List[BasePart]]

    is_error: bool = False