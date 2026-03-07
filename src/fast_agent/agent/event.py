from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Any, Literal, Dict, Optional, List
from fast_agent.llm import ToolCall, ToolResultMessage, LLMConfig, Context
from .snapshot import Snapshot
import time
import uuid
import asyncio


class EventChannelClosed(Exception):
    """EventChannel 已关闭，无法继续收发事件。"""
    pass

class BaseEventMetadata(BaseModel):
    """BaseEventMetadata 事件元数据基类"""
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))

class BaseEvent(BaseModel):
    """BaseEvent 事件基类"""
    id: str = Field(default_factory=lambda: f"event_{uuid.uuid4().hex[:16]}")

    type: str

    data: Any

    metadata: BaseEventMetadata = Field(default_factory=BaseEventMetadata)

# ===== 具体事件类型 =====

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


class InterruptEvent(BaseEvent):
    """InterruptEvent 中断事件"""
    type: Literal["error_interrupt", "human_review_interrupt"] = "error_interrupt"

    class InterruptEventData(BaseModel):
        reason: Optional[str] = None

        snapshot: Snapshot

    data: InterruptEventData

class HumanReviewEvent(BaseEvent):
    """HumanReviewEvent 人类审核事件"""
    type: Literal["human_review"] = "human_review"

    class HumanReviewEventData(BaseModel):
        content: Dict[str, Any]
        response_channel: asyncio.Future

    data: HumanReviewEventData

class HumanResponseEvent(BaseEvent):
    """HumanResponseEvent 人类审查响应事件"""
    # human_normal_response 表示正常响应（包括用户拒绝操作）
    # human_error_response 表示审查过程中发生错误（如客户端断联等）
    type: Literal["human_normal_response", "human_error_response"]

    class HumanResponseEventData(BaseModel):
        is_success: bool
        message: Optional[str] = None
        content: Optional[Dict[str, Any]] = None

    data: HumanResponseEventData


# ===== Event传输通道 =====
class EventChannel:
    """EventChannel 事件传输通道"""
    def __init__(self):
        self.event_queue = asyncio.Queue()
        self._close_sentinel = object()
        self._closed = False

    async def send_event(self, event: BaseEvent) -> None:
        """发送事件到通道"""
        if self._closed:
            raise EventChannelClosed("EventChannel is closed, cannot send event.")
        await self.event_queue.put(event)

    async def receive_event(self, timeout: Optional[int] = None) -> BaseEvent:
        """
        从通道接收事件（阻塞）
        
        参数说明：
        - timeout: 接收事件的超时时间（秒），默认为 None 表示无限等待
        """
        try:
            event = await asyncio.wait_for(self.event_queue.get(), timeout)
        except asyncio.TimeoutError:
            raise EventChannelClosed("EventChannel receive_event timeout.")

        if event is self._close_sentinel:
            self.event_queue.task_done()
            raise EventChannelClosed("EventChannel is closed.")

        return event

    def close(self) -> None:
        """关闭事件通道（幂等）。"""
        if self._closed:
            return
        self._closed = True
        self.event_queue.put_nowait(self._close_sentinel)

    @property
    def is_closed(self) -> bool:
        """事件通道是否已关闭。"""
        return self._closed
    
    def task_done(self):
        """标记事件处理完成"""
        self.event_queue.task_done()
