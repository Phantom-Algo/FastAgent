"""
Agent 有限状态机（FSM）驱动器模块

AgentFSM 是 BaseAgentFSM 的具体实现，负责：
1. 驱动状态流转：按 current_state → next_state 顺序执行各状态节点
2. 中断检查：在每个状态入口和每个事件产出后检查中断信号
3. 上下文回滚：中断时将共享数据恢复到当前状态执行前的检查点
4. 快照创建：中断时生成包含干净上下文的 Snapshot 对象
5. Guard 异常处理：捕获 ToolCallGuardTriggeredException，生成 Guard 快照

状态流转图：
    AfterUserInputState → LLMOutputState → AfterLLMOutputState
                                                │
                          ┌─────────────────────┤
                          │                     │
                 [tool_calls]             [stop / other]
                          │                     │
                BeforeExecuteToolsState   AfterFinishState → 终止
                          │
                ExecutingToolsState
                          │
                AfterExecuteToolsState → LLMOutputState (循环)
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass
from typing import AsyncGenerator, List, Optional

from ....types.agent.event.base_event import BaseEvent
from ....types.agent.fsm.base_agent_fsm import BaseAgentFSM
from ....types.agent.fsm.base_agent_fsm_state import BaseAgentFSMState
from ....types.agent.fsm.dto.agent_fsm_shared_data import AgentFSMSharedData
from ....types.agent.fsm.enum.agent_fsm_state_enum import AgentFSMStateEnum
from ....types.messages.domain.assistant_message import AssistantMessage, ToolCall
from ....types.messages.domain.tool_result_message import ToolResultMessage
from ....types.messages.domain.user_message import UserMessage
from ....types.agent.exceptions.tool_execution_interrupted_exception import ToolExecutionInterruptedException
from ....types.tool.domain.guard_triggered import ToolCallGuardTriggeredException

from ..event.events import GuardTriggeredEvent, InterruptEvent
from ..snapshot.snapshot import Snapshot


# ======= 中断信号 =======

@dataclass
class InterruptSignal:
    """
    中断信号，用于通知 FSM 停止当前流式输出。

    字段：
    - reason: 中断原因描述（如 "client_disconnect"、具体异常信息等）
    """
    reason: str = ""


class AgentFSM(BaseAgentFSM):
    """
    Agent 有限状态机驱动器。

    职责：
    1. 驱动状态流转：按 current_state → next_state 顺序执行各状态
    2. 中断检查：在每个状态入口和每个事件产出后检查中断信号
    3. 上下文回滚：中断时将共享数据恢复到当前状态执行前的检查点
    4. 快照创建：中断时生成包含干净上下文的 Snapshot 对象
    5. Guard 异常处理：捕获 ToolCallGuardTriggeredException 并生成 Guard 快照

    中断处理策略：
    - 客户端断联：外部调用 request_interrupt()，FSM 在检查点响应
    - 服务端异常：try/catch 捕获异常，回滚上下文并输出 InterruptEvent
    - Guard 触发：catch ToolCallGuardTriggeredException，生成 Guard 快照
    - 无论哪种中断，状态机立即终止，不再继续后续状态
    """

    def __init__(
        self,
        initial_state: BaseAgentFSMState,
        shared_data: AgentFSMSharedData,
    ):
        super().__init__(initial_state=initial_state, shared_data=shared_data)

        # 中断信号队列（线程安全）
        self._interrupt_queue: asyncio.Queue[InterruptSignal] = asyncio.Queue()

        # 状态入口检查点（用于中断时回滚共享数据）
        self._checkpoint_user_input: Optional[UserMessage] = None
        self._checkpoint_llm_output: Optional[AssistantMessage] = None
        self._checkpoint_tool_results: Optional[List[ToolResultMessage]] = None
        self._checkpoint_pending_tool_calls: Optional[List[ToolCall]] = None
        self._checkpoint_finished_tool_calls: Optional[List[ToolCall]] = None
        self._checkpoint_prebuilt_tool_results: Optional[List[ToolResultMessage]] = None
        self._has_checkpoint: bool = False

    # ===== 中断管理 =====

    def request_interrupt(self, reason: str = "client_disconnect") -> None:
        """
        请求中断当前流式输出。

        外部（如连接监控模块）调用此方法向 FSM 发送中断信号。
        FSM 会在下一个检查点（状态入口或事件产出后）响应中断。
        """
        self._interrupt_queue.put_nowait(InterruptSignal(reason=reason))

    def check_interrupt(self) -> Optional[InterruptSignal]:
        """非阻塞检查中断队列，返回中断信号或 None。"""
        try:
            return self._interrupt_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    # ===== 检查点与快照 =====

    def _save_checkpoint(self) -> None:
        """
        在状态执行前保存共享数据检查点。

        深拷贝关键字段，确保中断时能回滚到干净状态。
        """
        self._checkpoint_user_input = deepcopy(self.shared_data.user_input)
        self._checkpoint_llm_output = deepcopy(self.shared_data.llm_output)
        self._checkpoint_tool_results = deepcopy(self.shared_data.tool_results)
        self._checkpoint_pending_tool_calls = deepcopy(self.shared_data.pending_tool_calls)
        self._checkpoint_finished_tool_calls = deepcopy(self.shared_data.finished_tool_calls)
        self._checkpoint_prebuilt_tool_results = deepcopy(self.shared_data.prebuilt_tool_results)
        self._has_checkpoint = True

    def _rollback_to_checkpoint(self) -> None:
        """将共享数据回滚到最近的检查点。"""
        if not self._has_checkpoint:
            return

        self.shared_data.user_input = self._checkpoint_user_input
        self.shared_data.llm_output = self._checkpoint_llm_output
        self.shared_data.tool_results = self._checkpoint_tool_results
        self.shared_data.pending_tool_calls = self._checkpoint_pending_tool_calls
        self.shared_data.finished_tool_calls = self._checkpoint_finished_tool_calls
        self.shared_data.prebuilt_tool_results = self._checkpoint_prebuilt_tool_results

    def create_snapshot(self, status: AgentFSMStateEnum) -> Snapshot:
        """
        基于当前共享数据创建快照对象。

        优先使用检查点数据（确保快照中的数据未被当前状态部分修改）。
        """
        return Snapshot(
            llm_config=deepcopy(self.shared_data.llm_config),
            context=deepcopy(self.shared_data.context),
            lifespan_manager=deepcopy(self.shared_data.lifespan_manager),
            user_input=deepcopy(self.shared_data.user_input),
            llm_output=deepcopy(self.shared_data.llm_output),
            tool_results=deepcopy(self.shared_data.tool_results) if self.shared_data.tool_results else None,
            pending_tool_calls=deepcopy(self.shared_data.pending_tool_calls),
            finished_tool_calls=deepcopy(self.shared_data.finished_tool_calls),
            prebuilt_tool_results=deepcopy(self.shared_data.prebuilt_tool_results),
            state=status,
        )

    def _make_interrupt_event(self, signal: InterruptSignal) -> InterruptEvent:
        """
        构造中断事件：回滚数据 → 创建快照 → 包装为 InterruptEvent。
        """
        self._rollback_to_checkpoint()
        snapshot = self.create_snapshot(
            status=self.current_state.get_status() if self.current_state else AgentFSMStateEnum.AFTER_USER_INPUT
        )
        return InterruptEvent(
            data=InterruptEvent.InterruptEventData(
                reason=signal.reason,
                snapshot=snapshot,
            )
        )

    def _make_guard_triggered_event(
        self,
        exc: ToolCallGuardTriggeredException,
    ) -> GuardTriggeredEvent:
        """
        构造 Guard 触发事件：回滚数据 → 创建带 guard 上下文的快照。
        """
        self._rollback_to_checkpoint()

        snapshot = Snapshot(
            llm_config=deepcopy(self.shared_data.llm_config),
            context=deepcopy(self.shared_data.context),
            lifespan_manager=deepcopy(self.shared_data.lifespan_manager),
            user_input=deepcopy(self.shared_data.user_input),
            llm_output=deepcopy(self.shared_data.llm_output),
            tool_results=deepcopy(self.shared_data.tool_results) if self.shared_data.tool_results else None,
            tool_call_guard_triggered_contexts=exc.contexts,
            pending_tool_calls=deepcopy(self.shared_data.pending_tool_calls),
            prebuilt_tool_results=deepcopy(self.shared_data.prebuilt_tool_results),
            state=AgentFSMStateEnum.BEFORE_EXECUTE_TOOLS,
        )

        return GuardTriggeredEvent(
            data=GuardTriggeredEvent.GuardTriggeredEventData(
                guard_triggered_contexts=exc.contexts,
                snapshot=snapshot,
            )
        )

    # ===== 状态机主循环 =====

    async def run(self) -> AsyncGenerator[BaseEvent, None]:
        """
        驱动状态机执行，依次执行各状态并产出事件流。

        中断处理策略：
        - 客户端断联：外部调用 request_interrupt()，FSM 在检查点响应
        - Guard 触发：ToolCallGuardTriggeredException → 回滚并输出 GuardTriggeredEvent
        - 服务端异常：通用 Exception → 回滚并输出 InterruptEvent
        - 无论哪种中断，状态机立即终止
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

            except ToolCallGuardTriggeredException as guard_exc:
                # Guard 触发：生成 guard 快照并中断
                
                yield self._make_guard_triggered_event(guard_exc)
                return

            except ToolExecutionInterruptedException as tool_exc:
                # 工具执行阶段部分成功后中断：保留已完成进度，等待恢复继续执行失败项
                self._rollback_to_checkpoint()
                self.shared_data.pending_tool_calls = deepcopy(tool_exc.pending_tool_calls)
                self.shared_data.finished_tool_calls = deepcopy(tool_exc.finished_tool_calls)
                self.shared_data.tool_results = deepcopy(tool_exc.tool_results)
                self.shared_data.prebuilt_tool_results = None

                yield InterruptEvent(
                    data=InterruptEvent.InterruptEventData(
                        reason=f"tool_execution_interrupted: {str(tool_exc)}",
                        snapshot=self.create_snapshot(status=AgentFSMStateEnum.EXECUTING_TOOLS),
                    )
                )
                return

            except Exception as e:
                # 服务端异常：回滚并输出中断事件

                yield self._make_interrupt_event(
                    InterruptSignal(reason=f"server_error: {type(e).__name__}: {str(e)}")
                )
                return
