"""
Agent 有限状态机（FSM）核心模块

包含：
- InterruptSignal: 中断信号数据类
- IAgentState: 状态接口抽象基类
- AgentFSM: 状态机驱动器，负责状态流转、中断检查、快照创建与上下文回滚
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, AsyncGenerator, List
from copy import deepcopy
import asyncio
import logging

from .state import AgentState
from .snapshot import Snapshot
from .event import BaseEvent, InterruptEvent
from fast_agent.llm import UserMessage, AssistantMessage, ToolResultMessage

if TYPE_CHECKING:
    from .agent import Agent

logger = logging.getLogger(__name__)


# ======= 中断信号 =======

@dataclass
class InterruptSignal:
    """
    中断信号，用于通知 FSM 停止当前流式输出。

    字段：
    - reason: 中断原因描述（如 "client_disconnect"、具体异常信息等）
    """
    reason: str = ""


# ======= 状态接口 =======

class IAgentState(ABC):
    """
    Agent 状态接口（状态模式基类）

    所有具体状态类必须实现：
    - get_status(): 返回当前状态对应的 AgentState 枚举值
    - execute(fsm): 异步生成器，执行该状态的业务逻辑并产出事件流
    """

    @abstractmethod
    def get_status(self) -> AgentState:
        """返回当前状态对应的 AgentState 枚举值"""
        ...

    @abstractmethod
    async def execute(self, fsm: AgentFSM) -> AsyncGenerator[BaseEvent, None]:
        """
        执行当前状态的逻辑，产出事件流。

        约定：
        - 执行完毕后应设置 fsm.next_state 指向下一个状态
        - fsm.next_state = None 表示状态机终止
        - 通过 yield 产出的 BaseEvent 会逐个传递给调用方
        """
        ...


# ======= 状态机驱动器 =======

class AgentFSM:
    """
    AgentFSM Agent 有限状态机驱动器

    职责：
    1. 驱动状态流转：按 current_state → next_state 顺序执行各状态
    2. 中断检查：在每个状态入口和每个事件产出后检查中断信号
    3. 上下文回滚：中断时将 Agent 上下文恢复到当前状态执行前的检查点
    4. 快照创建：中断时生成包含干净上下文的 Snapshot 对象

    状态流转图：
        AfterUserInputState → LLMOutputState → AfterLLMOutputState
                                                     │
                              ┌──────────────────────┤
                              │                      │
                     [tool_calls]              [stop / other]
                              │                      │
                    BeforeExecuteToolsState    AfterFinishState → 终止
                              │
                    ExecutingToolsState
                              │
                    AfterExecuteToolsState → LLMOutputState (循环)
    """

    def __init__(
        self,
        agent: Agent,
        initial_state: IAgentState,
        user_input: Optional[UserMessage] = None,
        llm_output: Optional[AssistantMessage] = None,
        tool_results: Optional[List[ToolResultMessage]] = None,
    ):
        self.agent = agent
        self.current_state: Optional[IAgentState] = initial_state
        self.next_state: Optional[IAgentState] = None

        # ===== 跨状态共享的业务数据 =====
        self.user_input = user_input
        self.llm_output = llm_output
        self.tool_results = tool_results

        # ===== 中断信号队列（线程安全） =====
        self._interrupt_queue: asyncio.Queue[InterruptSignal] = asyncio.Queue()

        # ===== 状态入口检查点（用于中断时回滚上下文） =====
        self._checkpoint_llm_config = None
        self._checkpoint_context = None
        self._checkpoint_lifespan = None
        self._checkpoint_user_input = None
        self._checkpoint_llm_output = None
        self._checkpoint_tool_results = None
        self._has_checkpoint = False

    # ===== 中断管理 =====

    def request_interrupt(self, reason: str = "client_disconnect") -> None:
        """
        请求中断当前流式输出。

        外部（如连接监控模块）调用此方法向 FSM 发送中断信号。
        FSM 会在下一个检查点（状态入口或事件产出后）响应中断。
        """
        self._interrupt_queue.put_nowait(InterruptSignal(reason=reason))

    def check_interrupt(self) -> Optional[InterruptSignal]:
        """非阻塞检查中断队列，返回中断信号或 None"""
        try:
            return self._interrupt_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    # ===== 检查点与快照 =====

    def _save_checkpoint(self) -> None:
        """
        在状态执行前保存上下文检查点。

        深拷贝 context、llm_config、lifespan、user_input、llm_output、tool_results，确保中断时能回滚到干净状态。
        注：深拷贝开销与消息数量成正比，对于正常流程不构成瓶颈。
        """
        self._checkpoint_context = deepcopy(self.agent.context)
        self._checkpoint_llm_config = deepcopy(self.agent.llm_config)
        self._checkpoint_lifespan = deepcopy(self.agent.lifespan)
        self._checkpoint_user_input = deepcopy(self.user_input)
        self._checkpoint_llm_output = deepcopy(self.llm_output)
        self._checkpoint_tool_results = deepcopy(self.tool_results)
        self._has_checkpoint = True

    def _rollback_to_checkpoint(self) -> None:
        """将 Agent 上下文回滚到最近的检查点"""
        if not self._has_checkpoint:
            return

        self.agent.context = self._checkpoint_context
        self.agent.llm_config = self._checkpoint_llm_config
        self.agent.lifespan = self._checkpoint_lifespan
        self.user_input = self._checkpoint_user_input
        self.llm_output = self._checkpoint_llm_output
        self.tool_results = self._checkpoint_tool_results

    def create_snapshot(self, status: AgentState) -> Snapshot:
        """
        基于当前 FSM 状态创建快照对象。

        优先使用检查点上下文（确保快照中的上下文未被当前状态部分修改）。
        """
        return Snapshot(
            llm_config=deepcopy(self.agent.llm_config),
            context=deepcopy(self.agent.context),
            lifespan=deepcopy(self.agent.lifespan),
            user_input=deepcopy(self.user_input),
            llm_output=deepcopy(self.llm_output),
            tool_results=deepcopy(self.tool_results) if self.tool_results is not None else None,
            status=status,
        )

    def _make_interrupt_event(self, signal: InterruptSignal) -> InterruptEvent:
        """
        构造中断事件：回滚上下文 → 创建快照 → 包装为 InterruptEvent。

        参数：
        - signal: 触发中断的信号（包含中断原因）
        """
        self._rollback_to_checkpoint()
        snapshot = self.create_snapshot(
            status=self.current_state.get_status() if self.current_state else AgentState.AFTER_USER_INPUT
        )
        return InterruptEvent(
            data=InterruptEvent.InterruptEventData(
                reason=signal.reason,
                snapshot=snapshot,
            )
        )

    # ===== 状态机主循环 =====

    async def run(self) -> AsyncGenerator[BaseEvent, None]:
        """
        驱动状态机执行，依次执行各状态并产出事件流。

        中断处理策略：
        - 客户端断联：外部调用 request_interrupt()，FSM 在检查点响应
        - 服务端异常：try/catch 捕获异常，回滚上下文并输出 InterruptEvent
        - 无论哪种中断，状态机立即终止，不再继续后续状态
        """
        while self.current_state is not None:
            # ---- 状态入口：检查中断 ----
            interrupt = self.check_interrupt()
            if interrupt:
                yield self._make_interrupt_event(interrupt)
                return

            # ---- 保存检查点（用于中断时回滚） ----
            self._save_checkpoint()

            try:
                self.next_state = None

                async for event in self.current_state.execute(self):
                    yield event

                    # ---- 每个事件产出后检查中断 ----
                    interrupt = self.check_interrupt()
                    if interrupt:
                        yield self._make_interrupt_event(interrupt)
                        return

                # 状态执行完毕，切换到下一个状态
                self.current_state = self.next_state

            except Exception as e:
                # 服务端异常：回滚并输出中断事件
                logger.error(
                    "Agent FSM 状态 [%s] 执行异常: %s",
                    self.current_state.get_status().value if self.current_state else "unknown",
                    str(e),
                    exc_info=True,
                )
                yield self._make_interrupt_event(
                    InterruptSignal(reason=f"server_error: {type(e).__name__}: {str(e)}")
                )
                return
