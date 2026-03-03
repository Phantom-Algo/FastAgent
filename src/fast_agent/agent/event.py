from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Any, Literal, Dict, Optional, List
from fast_agent.llm import ToolCall, ToolResultMessage, LLMConfig, Context
import time
import uuid

class BaseEventMetadata(BaseModel):
    """BaseEventMetadata 事件元数据基类"""
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))

class BaseEvent(BaseModel):
    """BaseEvent 事件基类"""
    id: str = Field(default_factory=lambda: f"event_{uuid.uuid4().hex[:16]}")

    type: str

    data: Any

    metadata: BaseEventMetadata = Field(default_factory=BaseEventMetadata)


class AssistantMessageChunkOutputEvent(BaseEvent):
    """ChunkOutputEvent 分块输出事件"""
    type: Literal["chunk_output"] = "chunk_output"

    class AssistantMessageChunkOutputEventData(BaseModel):
        type: Literal["reasoning_content", "content", "refusal"]

        reasoning_content: Optional[str] = None

        content: Optional[str] = None

        refusal: Optional[str] = None

        @model_validator(mode='after')
        def check_at_least_one_field(self):
            if not any([self.reasoning_content, self.content, self.refusal]):
                raise ValueError("AssistantMessageChunkOutputEventData: At least one of 'reasoning_content', 'content', or 'refusal' must be provided.")
            return self

    data: AssistantMessageChunkOutputEventData

class ToolCallEvent(BaseEvent):
    """ToolCallEvent 工具调用事件"""
    type: Literal["tool_call"] = "tool_call"

    class ToolCallEventData(BaseModel):
        tool_call_id: str

        function_name: str

        function_args: Dict[str, Any]

    data: ToolCallEventData

class AssistantMessageOutputEvent(BaseEvent):
    """AssistantMessageOutputEvent LLM完整输出事件"""
    type: Literal["assistant_message_output"] = "assistant_message_output"

    class AssistantMessageOutputEventData(BaseModel):
        reasoning_content: Optional[str] = None

        content: Optional[str] = None

        refusal: Optional[str] = None

        tool_calls: Optional[List[ToolCall]] = None

        finish_reason: str = "unknown"

        token_usage: Optional[int] = None

        model: Optional[str] = None

    data: AssistantMessageOutputEventData

class RoundStopEvent(BaseEvent):
    """RoundStopEvent 轮次结束事件"""
    type: Literal["round_stop"] = "round_stop"

    class RoundStopEventData(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        finish_reason: Literal["unknown", "stop", "length", "tool_calls", "content_filter", "balance", "error"] = "unknown"

        llm_config: LLMConfig

        context: Context

        kwargs: Dict[str, Any]

    data: RoundStopEventData


class ToolsExecutedEvent(BaseEvent):
    """ToolsExecutedEvent 工具执行完毕事件"""
    type: Literal["tools_executed"] = "tools_executed"

    class ToolsExecutedEventData(BaseModel):
        tool_results: List[ToolResultMessage]

    data: ToolsExecutedEventData