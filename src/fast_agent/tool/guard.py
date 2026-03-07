from pydantic import BaseModel
from typing import Any, Type, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from fast_agent.llm import ToolCall
    from schema import BaseTool

class GuardTriggeredToolCallContext(BaseModel):
    """护栏被触发的工具调用上下文信息"""
    tool_call: ToolCall
    tool_info: BaseTool

class ToolCallGuardTriggered(Exception):
    """工具调用护栏触发"""
    def __init__(
        self,
        message: str,
        contexts: List[GuardTriggeredToolCallContext],
        finished_tool_calls: Optional[List[ToolCall]] = None,
    ):
        super().__init__(message)
        self.contexts = contexts
        self.finished_tool_calls = finished_tool_calls or []

class GuardRequestSchema(BaseModel):
    """工具调用护栏请求参数类型基类"""

class GuardInfo(BaseModel):
    """工具调用护栏描述信息"""
    description: Optional[Any] = None
    request_schema: Type[GuardRequestSchema]