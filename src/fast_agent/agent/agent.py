"""
Agent 核心模块

基于有限状态机（FSM）+ 状态模式驱动的 Agent 实现，支持：
- 流式输出（stream）
- 生命周期钩子（lifespan）
- 快照与中断恢复（snapshot & interrupt）
- 客户端断联检测（interrupt queue）
"""

from fast_agent.llm import (
    LLMConfig,
    Context,
    UserMessage,
    AssistantMessageChunk,
    ToolCall,
    AssistantMessage,
    ToolResultMessage,
)
from .lifespan import Lifespan
from .fsm import AgentFSM
from .states import (
    IAgentState,
    AfterUserInputState,
    LLMOutputState,
    AfterLLMOutputState,
    BeforeExecuteToolsState,
    ExecutingToolsState,
    AfterExecuteToolsState,
    AfterFinishState
)
from .state import AgentState
from .snapshot import Snapshot
from .event import (
    BaseEvent,
    AssistantMessageChunkOutputEvent,
    ToolCallEvent,
    AssistantMessageOutputEvent,
    RoundStopEvent,
    ToolsExecutedEvent,
    InterruptEvent,
    EventChannel,
    EventChannelClosed,
)
from typing import Optional, AsyncGenerator
from copy import deepcopy
import asyncio


class Agent:
    """
    Agent 核心类

    基于有限状态机（FSM）驱动，将流式输出的各阶段拆分为独立的状态类：
    - 每个状态负责自身的业务逻辑和生命周期钩子调用
    - 状态机在每个检查点（状态入口、事件产出后）检查中断信号
    - 中断时自动回滚上下文、创建快照、发出 InterruptEvent

    使用方式：
        agent = Agent(llm_config=..., context=..., lifespan=...)
        async for event in agent.stream(user_input):
            # 处理事件
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        context: Context,
        lifespan: Lifespan = Lifespan(),
    ):
        self.llm_config = llm_config
        self.context = context
        self.lifespan = lifespan

        # 当前正在运行的 FSM 实例（用于外部中断控制）
        self._current_fsm: Optional[AgentFSM] = None

    # ===== 流式输出 API =====

    async def stream(self, user_input: UserMessage) -> AsyncGenerator[BaseEvent, None]:
        """
        Agent 流式输出入口

        创建状态机并从 AfterUserInputState 开始执行，逐个产出事件流。
        支持在运行期间通过 request_interrupt() 进行中断。
        """
        # 每次 stream 都使用全新通道
        self.event_channel = EventChannel()

        fsm = AgentFSM(
            agent=self,
            initial_state=AfterUserInputState(),
            user_input=user_input,
        )
        async for event in self._stream_unified_channel(fsm):
            yield event

    async def resume_stream(self, snapshot: Snapshot) -> AsyncGenerator[BaseEvent, None]:
        """
        恢复流式输出接口，基于传入的 Snapshot 恢复 Agent 状态并继续流式输出。

        参数列表：
        - snapshot: Snapshot 包含之前 Agent 状态的快照对象
        """
        # 恢复 Agent 状态
        self.llm_config = snapshot.llm_config
        self.context = snapshot.context
        self.lifespan = snapshot.lifespan

        # run-scope 事件通道：每次 resume_stream 都使用全新通道
        self.event_channel = EventChannel()

        # 从 snapshot 中恢复 FSM
        fsm = AgentFSM(
            agent=self,
            initial_state=self._get_initial_state_by_snapshot(snapshot),
            user_input=snapshot.user_input,
            llm_output=snapshot.llm_output,
            tool_results=snapshot.tool_results,
        )

        async for event in self._stream_unified_channel(fsm):
            yield event

    async def _stream_unified_channel(self, fsm: AgentFSM) -> AsyncGenerator[BaseEvent, None]:
        """统一执行 FSM 与 EventChannel 事件合流输出。"""
        self._current_fsm = fsm

        output_queue: asyncio.Queue = asyncio.Queue()
        fsm_done = False
        channel_done = False
        fsm_done_marker = object()
        channel_done_marker = object()

        async def _fsm_pump() -> None:
            nonlocal fsm_done
            try:
                async for event in fsm.run():
                    await output_queue.put(event)
            finally:
                fsm_done = True
                self.event_channel.close()
                await output_queue.put(fsm_done_marker)

        async def _channel_pump() -> None:
            nonlocal channel_done
            try:
                while True:
                    event = await self.event_channel.receive_event()
                    await output_queue.put(event)
                    self.event_channel.task_done()
            except EventChannelClosed:
                pass
            finally:
                channel_done = True
                await output_queue.put(channel_done_marker)

        fsm_task = asyncio.create_task(_fsm_pump())
        channel_task = asyncio.create_task(_channel_pump())

        try:
            while True:
                item = await output_queue.get()

                if item is fsm_done_marker or item is channel_done_marker:
                    if fsm_done and channel_done and output_queue.empty():
                        break
                    continue

                yield item
        finally:
            self.event_channel.close()

            for task in (fsm_task, channel_task):
                if not task.done():
                    task.cancel()

            # 等待两个 pump 任务结束，确保资源清理完毕
            await asyncio.gather(fsm_task, channel_task, return_exceptions=True)
            self._current_fsm = None

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
        """当前是否有正在执行的流式输出"""
        return self._current_fsm is not None

    # ===== Lifespan API =====

    def register_lifespan(self, lifespan: Lifespan) -> None:
        """
        注册生命周期方法

        参数列表：
        - lifespan: 用户传入的生命周期注册类实例，包含用户自定义实现的生命周期方法
        """
        self.lifespan = lifespan if lifespan is not None else Lifespan()

    def unregister_lifespan(self) -> None:
        """注销生命周期方法，重置为默认实现"""
        self.lifespan = Lifespan()

    def get_lifespan(self) -> Lifespan:
        """
        获取当前生命周期注册类实例

        返回值：
        - Lifespan: 当前生命周期注册类实例
        """
        return self.lifespan

    def update_lifespan(self, lifespan: Lifespan) -> None:
        """
        更新生命周期方法，允许在 Agent 运行过程中动态修改生命周期方法的实现

        参数列表：
        - lifespan: 用户传入的生命周期注册类实例，包含用户自定义实现的生命周期方法
        """
        self.register_lifespan(lifespan=lifespan)

    def update_lifespan_kwargs(self, kwargs: dict) -> None:
        """
        更新生命周期方法的扩展参数，允许在 Agent 运行过程中动态修改

        参数列表：
        - kwargs: 包含生命周期方法扩展参数的字典
        """
        if self.lifespan is None:
            self.lifespan = Lifespan()
        self.lifespan.kwargs.update(kwargs)

    def set_lifespan_kwargs(self) -> None:
        """重置生命周期方法的扩展参数"""
        if self.lifespan is None:
            self.lifespan = Lifespan()
        self.lifespan.kwargs = {}

    # ===== 工具函数 =====

    def _wrap_to_event(self, output) -> BaseEvent:
        """将 Adapter 的输出包装成 Agent 输出的事件对象"""
        if isinstance(output, AssistantMessageChunk):
            # 兼容三种类型字段，优先级 reasoning_content > content > refusal
            data = {}
            if output.reasoning_content_delta is not None:
                data["type"] = "reasoning_content"
                data["reasoning_content"] = output.reasoning_content_delta
            elif output.content_delta is not None:
                data["type"] = "content"
                data["content"] = output.content_delta
            elif output.refusal_delta is not None:
                data["type"] = "refusal"
                data["refusal"] = output.refusal_delta
            else:
                raise ValueError("AssistantMessageChunk: missing valid content field.")
            return AssistantMessageChunkOutputEvent(
                data=AssistantMessageChunkOutputEvent.AssistantMessageChunkOutputEventData(**data),
            )

        elif isinstance(output, ToolCall):
            return ToolCallEvent(
                data=ToolCallEvent.ToolCallEventData(
                    tool_call_id=getattr(output, "tool_call_id", None),
                    function_name=getattr(output, "function_name", None),
                    function_args=getattr(output, "function_args", {}),
                )
            )

        elif isinstance(output, AssistantMessage):
            return AssistantMessageOutputEvent(
                data=AssistantMessageOutputEvent.AssistantMessageOutputEventData(
                    reasoning_content=getattr(output, "reasoning_content", None),
                    content=getattr(output, "content", None),
                    refusal=getattr(output, "refusal", None),
                    tool_calls=getattr(output, "tool_calls", None),
                    finish_reason=getattr(output, "finish_reason", "unknown"),
                    token_usage=getattr(output, "token_usage", None),
                    model=getattr(output, "model", None),
                )
            )

        else:
            raise ValueError(f"Unsupported output type: {type(output)}")
        
    def _get_initial_state_by_snapshot(self, snapshot: Snapshot) -> IAgentState:
        """根据 Snapshot 中的状态信息返回对应的 FSM 初始状态实例"""
        status = snapshot.status
        if status == AgentState.AFTER_USER_INPUT:
            return AfterUserInputState()
        elif status == AgentState.LLM_OUTPUT:
            return LLMOutputState()
        elif status == AgentState.AFTER_LLM_OUTPUT:
            return AfterLLMOutputState()
        elif status == AgentState.BEFORE_EXECUTE_TOOLS:
            return BeforeExecuteToolsState()
        elif status == AgentState.EXECUTING_TOOLS:
            return ExecutingToolsState()
        elif status == AgentState.AFTER_EXECUTE_TOOLS:
            return AfterExecuteToolsState()
        elif status == AgentState.AFTER_FINISH:
            return AfterFinishState()
        else:
            raise ValueError(f"Unsupported snapshot status: {status}")