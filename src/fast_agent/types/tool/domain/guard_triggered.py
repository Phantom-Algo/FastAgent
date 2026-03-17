from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, ConfigDict

from ...messages.domain.assistant_message import ToolCall
from ...tool.base_tool import BaseTool



class ToolCallGuardTriggeredContext(BaseModel):
    """护栏被触发时的工具调用上下文。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool_call: ToolCall

    tool_info: Optional[BaseTool] = None


class ToolCallGuardTriggeredException(Exception):
    """工具调用护栏触发异常。"""

    def __init__(
        self,
        message: str,
        contexts: List[ToolCallGuardTriggeredContext],
        
    ):
        super().__init__(message)
        self.contexts = contexts