"""
Agent 核心模块

基于有限状态机（FSM）+ 状态模式驱动的 Agent 实现，支持：
- 流式输出（stream）
- 生命周期钩子（lifespan）
- 快照与中断恢复（snapshot & interrupt）
- Guard 机制（guard_policy → 人工审批 → 恢复执行）
- 客户端断联检测（interrupt queue）
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import AsyncGenerator, Dict, Literal, Optional, Union

from ...types.agent.base_agent import BaseAgent
from ...types.agent.event.base_event import BaseEvent
from ...types.agent.fsm.dto.agent_fsm_shared_data import AgentFSMSharedData
from ...types.agent.fsm.enum.agent_fsm_state_enum import AgentFSMStateEnum
from ...types.agent.snapshot.base_snapshot import BaseSnapshot
from ...types.context.base_context import BaseContext
from ...types.llm.base_llm_config import BaseLLMConfig
from ...types.messages.domain.assistant_message import AssistantMessage, ToolCall
from ...types.messages.domain.assistant_message_chunk import AssistantMessageChunk
from ...types.messages.domain.user_message import UserMessage
from ...types.tool.domain.guard_policy import GuardPolicyHumanResponseSchema

from .event.event_channel import EventChannel
from .event.events import (
    AssistantMessageChunkOutputEvent,
    AssistantMessageOutputEvent,
    ToolCallEvent,
)
from .lifespan.lifespan_manager import LifespanManager
from .fsm.agent_fsm import AgentFSM
from .fsm.states import (
    AfterExecuteToolsState,
    AfterFinishState,
    AfterLLMOutputState,
    AfterUserInputState,
    BeforeExecuteToolsState,
    ExecutingToolsState,
    LLMOutputState,
)

logger = logging.getLogger(__name__)


class Agent(BaseAgent):
    """
    Agent 核心类

    基于有限状态机（FSM）驱动，将流式输出的各阶段拆分为独立的状态类：
    - 每个状态负责自身的业务逻辑和生命周期钩子调用
    - 状态机在每个检查点（状态入口、事件产出后）检查中断信号
    - 中断时自动回滚上下文、创建快照、发出 InterruptEvent
    - Guard 触发时发出 GuardTriggeredEvent，客户端需收集 human_response 后调用 resume_stream

    使用方式：
        agent = Agent(llm_config=..., context=..., lifespan_manager=...)
        async for event in agent.stream(user_input):
            # 处理事件
    """

    def __init__(
        self,
        llm_config: BaseLLMConfig,
        context: BaseContext,
        lifespan_manager: Optional[LifespanManager] = None,
    ):
        self.llm_config = llm_config
        self.context = context
        self.lifespan_manager = lifespan_manager if lifespan_manager is not None else LifespanManager()

        # 当前正在运行的 FSM 实例（用于外部中断控制）
        self._current_fsm: Optional[AgentFSM] = None

    # ===== 流式输出 API =====

    async def stream(
        self,
        user_input: Union[str, UserMessage],
        stream_mode: Literal["chunk", "message"] = "chunk",
    ) -> AsyncGenerator[BaseEvent, None]:
        """
        Agent 流式输出入口。

        创建状态机并从 AfterUserInputState 开始执行，逐个产出事件流。
        支持在运行期间通过 request_interrupt() 进行中断。

        参数：
        - user_input: 用户输入消息
        - stream_mode: 流式输出模式（"chunk" 产出 chunk 级事件，"message" 仅产出完整消息事件）
        """
        # 创建事件通道
        event_channel = EventChannel()

        # 将事件包装器注入到 kwargs 中，供 LLMOutputState 使用
        kwargs = self.lifespan_manager.get_kwargs()
        kwargs["_wrap_to_event"] = self._wrap_to_event
        kwargs["_stream_mode"] = stream_mode

        # 构建共享数据
        if isinstance(user_input, str):
            user_input = UserMessage(content=user_input)
        
        shared_data = AgentFSMSharedData(
            llm_config=self.llm_config,
            context=self.context,
            lifespan_manager=self.lifespan_manager,
            event_channel=event_channel,
            user_input=user_input,
        )

        # 创建并运行 FSM
        fsm = AgentFSM(
            initial_state=AfterUserInputState(),
            shared_data=shared_data,
        )
        self._current_fsm = fsm

        try:
            async for event in fsm.run():
                yield event
        finally:
            self._current_fsm = None
            event_channel.close()

    async def resume_stream(
        self,
        snapshot: BaseSnapshot,
        human_response: Optional[Dict[str, GuardPolicyHumanResponseSchema]] = None,
        stream_mode: Literal["chunk", "message"] = "chunk",
    ) -> AsyncGenerator[BaseEvent, None]:
        """
        恢复流式输出接口，基于传入的 Snapshot 恢复 Agent 状态并继续流式输出。

        参数：
        - snapshot: 包含之前 Agent 状态的快照对象
        - human_response: Guard 触发后用户提供的人工审批响应，key 为 tool_call_id
        - stream_mode: 流式输出模式
        """
        # 从 snapshot 恢复 Agent 状态
        self.llm_config = deepcopy(snapshot.llm_config)
        self.context = deepcopy(snapshot.context)
        self.lifespan_manager = deepcopy(snapshot.lifespan_manager)

        # 创建事件通道
        event_channel = EventChannel()

        # 将事件包装器注入到 kwargs 中
        kwargs = self.lifespan_manager.get_kwargs()
        kwargs["_wrap_to_event"] = self._wrap_to_event
        kwargs["_stream_mode"] = stream_mode

        # 构建共享数据
        shared_data = AgentFSMSharedData(
            llm_config=self.llm_config,
            context=self.context,
            lifespan_manager=self.lifespan_manager,
            event_channel=event_channel,
            user_input=deepcopy(snapshot.user_input),
            llm_output=deepcopy(snapshot.llm_output),
            tool_results=deepcopy(snapshot.tool_results) if snapshot.tool_results else None,
            human_response=human_response,
            pending_tool_calls=deepcopy(snapshot.pending_tool_calls) if snapshot.pending_tool_calls else None,
            finished_tool_calls=deepcopy(snapshot.finished_tool_calls) if snapshot.finished_tool_calls else None,
            prebuilt_tool_results=deepcopy(snapshot.prebuilt_tool_results) if snapshot.prebuilt_tool_results else None,
        )

        # 根据 snapshot 中的状态确定恢复的初始状态
        initial_state = self._get_initial_state_by_snapshot(snapshot)

        # 创建并运行 FSM
        fsm = AgentFSM(
            initial_state=initial_state,
            shared_data=shared_data,
        )
        self._current_fsm = fsm

        try:
            async for event in fsm.run():
                yield event
        finally:
            self._current_fsm = None
            event_channel.close()

    # ===== 中断控制 API =====

    def request_interrupt(self, reason: str = "client_disconnect") -> None:
        """
        请求中断当前正在执行的流式输出。

        适用场景：
        - 客户端与服务端断联时，由连接监控调用
        - 需要主动终止当前 Agent 轮次时

        中断后 Agent 会在下一个检查点：
        1. 回滚当前阶段的上下文修改
        2. 创建包含当前状态的 Snapshot
        3. 产出 InterruptEvent 事件
        4. 终止流式输出
        """
        if self._current_fsm is not None:
            self._current_fsm.request_interrupt(reason=reason)

    @property
    def is_running(self) -> bool:
        """当前是否有正在执行的流式输出。"""
        return self._current_fsm is not None

    # ===== Lifespan API =====

    def register_lifespan(self, lifespan_manager: LifespanManager) -> None:
        """注册生命周期管理器。"""
        self.lifespan_manager = lifespan_manager if lifespan_manager is not None else LifespanManager()

    def unregister_lifespan(self) -> None:
        """注销生命周期管理器，重置为默认实现。"""
        self.lifespan_manager = LifespanManager()

    def get_lifespan(self) -> LifespanManager:
        """获取当前生命周期管理器实例。"""
        return self.lifespan_manager

    def update_lifespan(self, lifespan_manager: LifespanManager) -> None:
        """更新生命周期管理器，允许在 Agent 运行过程中动态修改。"""
        self.register_lifespan(lifespan_manager=lifespan_manager)

    def update_lifespan_kwargs(self, kwargs: dict) -> None:
        """更新生命周期管理器的 kwargs。"""
        self.lifespan_manager.update_kwargs(kwargs)

    # ===== 工具函数 =====

    def _wrap_to_event(self, output) -> BaseEvent:
        """将 Adapter 的输出包装成 Agent 事件对象。"""
        if isinstance(output, AssistantMessageChunk):
            # 兼容三种类型字段，优先级 reasoning_content > content > refusal
            data = {}
            if output.reasoning_content_delta is not None:
                data["chunk_type"] = "reasoning_content"
                data["reasoning_content"] = output.reasoning_content_delta
            elif output.content_delta is not None:
                data["chunk_type"] = "content"
                data["content"] = output.content_delta
            elif output.refusal_delta is not None:
                data["chunk_type"] = "refusal"
                data["refusal"] = output.refusal_delta
            else:
                raise ValueError("AssistantMessageChunk: missing valid content field.")
            return AssistantMessageChunkOutputEvent(
                data=AssistantMessageChunkOutputEvent.AssistantMessageChunkOutputEventData(**data),
            )

        elif isinstance(output, ToolCall):
            return ToolCallEvent(
                data=ToolCallEvent.ToolCallEventData(
                    tool_call_id=output.tool_call_id,
                    function_name=output.function_name,
                    function_args=output.function_args,
                )
            )

        elif isinstance(output, AssistantMessage):
            return AssistantMessageOutputEvent(
                data=AssistantMessageOutputEvent.AssistantMessageOutputEventData(
                    reasoning_content=output.reasoning_content,
                    content=output.content,
                    refusal=output.refusal,
                    tool_calls=output.tool_calls,
                    finish_reason=output.finish_reason,
                    token_usage=output.token_usage,
                    model=output.model,
                )
            )

        else:
            raise ValueError(f"Unsupported adapter output type: {type(output)}")

    def _get_initial_state_by_snapshot(self, snapshot: BaseSnapshot):
        """根据 Snapshot 中的状态信息返回对应的 FSM 初始状态实例。"""
        status = snapshot.state
        _state_map = {
            AgentFSMStateEnum.AFTER_USER_INPUT: AfterUserInputState,
            AgentFSMStateEnum.LLM_OUTPUT: LLMOutputState,
            AgentFSMStateEnum.AFTER_LLM_OUTPUT: AfterLLMOutputState,
            AgentFSMStateEnum.BEFORE_EXECUTE_TOOLS: BeforeExecuteToolsState,
            AgentFSMStateEnum.EXECUTING_TOOLS: ExecutingToolsState,
            AgentFSMStateEnum.AFTER_EXECUTE_TOOLS: AfterExecuteToolsState,
            AgentFSMStateEnum.AFTER_FINISH: AfterFinishState,
        }
        state_cls = _state_map.get(status)
        if state_cls is None:
            raise ValueError(f"Unsupported snapshot state: {status}")
        return state_cls()
