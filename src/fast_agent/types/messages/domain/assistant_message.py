from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional, Union, List, Dict, Any
import uuid
from ..base_message import BaseMessage
from enum import Enum

class AssistantMessageFinishReasonEnum(str, Enum):
    UNKNOWN = "unknown"
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    BALANCE = "balance"
    ERROR = "error"

class ToolCall(BaseModel):
    """ToolCall 大模型工具调用请求类"""
    type: Literal['tool_call'] = 'tool_call'

    # 工具调用唯一 ID，用于与后续的工具调用结果进行关联（为兼容无ID的情况，设置自动生成）
    tool_call_id: str = Field(default_factory=lambda: f"call_{str(uuid.uuid4().hex[:16])}")

    function_name: str
    function_args: Dict[str, Any]

class AssistantMessage(BaseMessage):
    """AssistantMessage AI 消息类"""
    id: str = Field(default_factory=lambda: f"assistant_{str(uuid.uuid4().hex[:16])}")

    type: Literal['assistant_message'] = 'assistant_message'

    role: Literal['assistant'] = 'assistant'

    reasoning_content: Optional[str] = None

    content: Optional[str] = None

    tool_calls: Optional[List[ToolCall]] = None

    refusal: Optional[str] = None

    finish_reason: AssistantMessageFinishReasonEnum = AssistantMessageFinishReasonEnum.UNKNOWN

    token_usage: Optional[int] = None

    model: Optional[str] = None

    @model_validator(mode='after')
    def check_content_or_tools(self):
        """确保消息中至少有文本或者工具调用之一，避免空消息"""
        if not self.reasoning_content and not self.content and not self.tool_calls and not self.refusal:
            raise ValueError("AssistantMessage must contain 'reasoning_content', 'content', 'tool_calls', or 'refusal'")
        return self