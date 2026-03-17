from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, model_validator

from ....types.agent.event.base_event import BaseEvent
from ....types.agent.enum.agent_round_stop_enum import AgentRoundStopEnum
from ....types.messages.domain.assistant_message import ToolCall, AssistantMessageFinishReasonEnum
from ....types.messages.domain.tool_result_message import ToolResultMessage
from ....types.tool.domain.guard_triggered import ToolCallGuardTriggeredContext
from ....types.agent.snapshot.base_snapshot import BaseSnapshot
from ....types.llm.base_llm_config import BaseLLMConfig
from ....types.context.base_context import BaseContext


# ================================================================
# AskHuman 相关事件
# ================================================================

class AskHumanEvent(BaseEvent):
    """人工交互请求事件，由工具执行中的 ask_human 机制触发。"""

    type: Literal["ask_human_event"] = "ask_human_event"

    class AskHumanEventData(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        content: Dict[str, Any]

        ask_human_response_channel: asyncio.Future

    data: AskHumanEventData


class AskHumanResponseEvent(BaseEvent):
    """人工交互响应事件，携带人类审批结果。"""

    type: Literal["ask_human_response_event"] = "ask_human_response_event"

    class AskHumanResponseEventData(BaseModel):
        response_success: bool

        message: str

        response_content: Dict[str, Any]

    data: AskHumanResponseEventData


# ================================================================
# 流式输出事件
# ================================================================

class AssistantMessageChunkOutputEvent(BaseEvent):
    """LLM 流式 chunk 输出事件，每次携带一个文本片段（reasoning / content / refusal 三选一）。"""

    type: Literal["chunk_output_event"] = "chunk_output_event"

    class AssistantMessageChunkOutputEventData(BaseModel):
        """chunk 数据，三个可选字段至少有一个非空。"""
        chunk_type: Literal["reasoning_content", "content", "refusal"]

        reasoning_content: Optional[str] = None

        content: Optional[str] = None

        refusal: Optional[str] = None

        @model_validator(mode="after")
        def _check_at_least_one(self):
            if not any([self.reasoning_content, self.content, self.refusal]):
                raise ValueError(
                    "AssistantMessageChunkOutputEventData: at least one of "
                    "'reasoning_content', 'content', or 'refusal' must be provided."
                )
            return self

    data: AssistantMessageChunkOutputEventData


class AssistantMessageOutputEvent(BaseEvent):
    """LLM 完整输出事件，在流式输出结束后发出，包含完整的 AssistantMessage 信息。"""

    type: Literal["assistant_message_output_event"] = "assistant_message_output_event"

    class AssistantMessageOutputEventData(BaseModel):
        reasoning_content: Optional[str] = None

        content: Optional[str] = None

        refusal: Optional[str] = None

        tool_calls: Optional[List[ToolCall]] = None

        finish_reason: AssistantMessageFinishReasonEnum = AssistantMessageFinishReasonEnum.UNKNOWN

        token_usage: Optional[int] = None

        model: Optional[str] = None

    data: AssistantMessageOutputEventData


# ================================================================
# 工具调用事件
# ================================================================

class ToolCallEvent(BaseEvent):
    """工具调用检测事件，LLM 输出中每检测到一个 tool_call 即产出一个。"""

    type: Literal["tool_call_event"] = "tool_call_event"

    class ToolCallEventData(BaseModel):
        tool_call_id: str

        function_name: str

        function_args: Dict[str, Any]

    data: ToolCallEventData


class ToolsExecutedEvent(BaseEvent):
    """工具执行完毕事件，携带所有工具的执行结果。"""

    type: Literal["tools_executed_event"] = "tools_executed_event"

    class ToolsExecutedEventData(BaseModel):
        tool_results: List[ToolResultMessage]

    data: ToolsExecutedEventData


# ================================================================
# 控制事件
# ================================================================

class RoundStopEvent(BaseEvent):
    """轮次结束事件，标志着一轮 Agent 交互的终止。"""

    type: Literal["round_stop_event"] = "round_stop_event"

    class RoundStopEventData(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        finish_reason: AgentRoundStopEnum = AgentRoundStopEnum.UNKNOWN

        llm_config: BaseLLMConfig = None 

        context: BaseContext = None  

        kwargs: Dict[str, Any] = {}

    data: RoundStopEventData


class InterruptEvent(BaseEvent):
    """
    中断事件，在以下场景产出：
    - 外部调用 request_interrupt()（客户端断联等）
    - FSM 执行中遇到服务端异常
    携带 reason 和可用于恢复的 Snapshot。
    """

    type: Literal["interrupt_event"] = "interrupt_event"

    class InterruptEventData(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        reason: Optional[str] = None

        snapshot: BaseSnapshot = None  

    data: InterruptEventData


class GuardTriggeredEvent(BaseEvent):
    """
    Guard 触发事件，在 BeforeExecuteTools 阶段检测到带 GuardPolicy 的工具调用时产出。
    携带 guard 上下文和可用于恢复的 Snapshot，客户端需收集 human_response 后调用 resume_stream。
    """

    type: Literal["guard_triggered_event"] = "guard_triggered_event"

    class GuardTriggeredEventData(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        guard_triggered_contexts: List[ToolCallGuardTriggeredContext]

        snapshot: BaseSnapshot = None 

    data: GuardTriggeredEventData

