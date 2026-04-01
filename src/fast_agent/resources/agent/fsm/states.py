"""
Agent 状态模式实现模块

定义了 Agent 状态机中各阶段的具体逻辑，每个状态类负责：
1. 执行该阶段的业务逻辑（生命周期钩子调用、LLM 流式输出、工具执行等）
2. 产出该阶段的事件（BaseEvent 子类）
3. 决定下一个状态的流转（设置 fsm.next_state）

状态流转顺序：
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

from typing import AsyncGenerator, List

from ....types.agent.event.base_event import BaseEvent
from ....types.agent.fsm.base_agent_fsm_state import BaseAgentFSMState
from ....types.agent.fsm.enum.agent_fsm_state_enum import AgentFSMStateEnum
from ....types.agent.lifespan.dto.lifespan_dto import (
    AfterExecuteToolsRequest,
    AfterExecuteToolsResponse,
    AfterFinishRequest,
    AfterFinishResponse,
    AfterLLMOutputRequest,
    AfterLLMOutputResponse,
    AfterUserInputRequest,
    AfterUserInputResponse,
    BeforeExecuteToolsRequest,
    BeforeExecuteToolsResponse,
    ExecutingToolsRequest,
    ExecutingToolsResponse,
)
from ....types.messages.domain.assistant_message import AssistantMessage

from ...adapter.adapter_factory import AdapterFactory
from ..event.events import RoundStopEvent, ToolsExecutedEvent

# 使用 TYPE_CHECKING 避免循环导入
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_fsm import AgentFSM


# ======= 用户输入处理状态 =======

class AfterUserInputState(BaseAgentFSMState):
    """
    用户输入处理状态

    执行逻辑：
    1. 将用户输入添加到原始消息列表（raw_messages）
    2. 调用 after_user_input 生命周期钩子
    3. 将处理后的用户输入添加到工作消息列表（work_messages）
    4. 流转到 LLMOutputState
    """

    def get_status(self) -> AgentFSMStateEnum:
        return AgentFSMStateEnum.AFTER_USER_INPUT

    async def execute(self, fsm: "AgentFSM") -> AsyncGenerator[BaseEvent, None]:
        sd = fsm.shared_data
        user_input = sd.user_input

        # 将用户原始输入存入 raw_messages
        sd.context.add_raw_message(user_input)

        # 调用 after_user_input 生命周期钩子
        handler = sd.lifespan_manager.get_lifespan("after_user_input")
        request = AfterUserInputRequest(
            llm_config=sd.llm_config,
            context=sd.context,
            event_channel=sd.event_channel,
            user_input=user_input,
            kwargs=sd.lifespan_manager.get_kwargs(),
        )
        response: AfterUserInputResponse = await handler.execute(request)

        # 回写 kwargs 和上下文
        sd.lifespan_manager.set_kwargs(response.kwargs)
        sd.context = response.context
        sd.llm_config = response.llm_config

        # 更新共享数据：将处理后的用户输入存入 work_messages
        sd.user_input = response.user_input
        sd.context.add_work_message(sd.user_input)

        # 流转到 LLM 输出状态
        fsm.next_state = LLMOutputState()
        return
        yield  # 使函数成为异步生成器（本状态无事件产出）


# ======= LLM 流式输出状态 =======

class LLMOutputState(BaseAgentFSMState):
    """
    LLM 流式输出状态

    执行逻辑：
    1. 根据 llm_config 获取对应的 Adapter
    2. 调用 Adapter.stream() 进行流式输出
    3. 通过 Agent._wrap_to_event() 将 Adapter 输出包装为事件并逐个产出
    4. 接收到完整的 AssistantMessage 后流转到 AfterLLMOutputState
    """

    def get_status(self) -> AgentFSMStateEnum:
        return AgentFSMStateEnum.LLM_OUTPUT

    async def execute(self, fsm: "AgentFSM") -> AsyncGenerator[BaseEvent, None]:
        sd = fsm.shared_data

        # 根据 llm_config 获取 Adapter
        adapter_cls = AdapterFactory.get_adapter_cls(provider=sd.llm_config.provider)
        adapter = adapter_cls()

        # 获取事件包装器（从 kwargs 中取得，由 Agent 注入）
        wrap_to_event = sd.lifespan_manager.get_kwargs().get("_wrap_to_event")

        # 获取流式输出模式（从 kwargs 中取得，由 Agent 注入）
        stream_mode = sd.lifespan_manager.get_kwargs().get("_stream_mode")

        llm_output = None
        # 调用 Adapter 流式输出，逐个产出事件
        if stream_mode == "chunk":
            async for output in adapter.stream(
                llm_config=sd.llm_config,
                context=sd.context,
            ):
                # 将 Adapter 输出包装为事件
                if wrap_to_event is not None:
                    event = wrap_to_event(output)
                    yield event

                # 如果收到完整的 AssistantMessage，说明流式输出结束
                if isinstance(output, AssistantMessage):
                    llm_output = output
                    break
        # 调用 Adapter 非流式输出，直接获取完整的 AssistantMessage
        elif stream_mode == "message":
            llm_output = await adapter.invoke(
                llm_config=sd.llm_config,
                context=sd.context,
            )
            if wrap_to_event is not None:
                yield wrap_to_event(llm_output)

        if llm_output is None:
            raise ValueError("LLM streaming terminated abnormally: no AssistantMessage received.")

        # 更新共享数据，流转到 after_llm_output 状态
        sd.llm_output = llm_output
        fsm.next_state = AfterLLMOutputState()


# ======= LLM 输出后处理与路由状态 =======

class AfterLLMOutputState(BaseAgentFSMState):
    """
    LLM 输出后处理与路由状态

    执行逻辑：
    1. 将 LLM 输出添加到消息列表（raw_messages + work_messages）
    2. 调用 after_llm_output 生命周期钩子
    3. 根据 finish_reason 进行路由决策：
       - "tool_calls" → BeforeExecuteToolsState
       - "stop"       → AfterFinishState
       - 其他         → 产出 RoundStopEvent 并终止状态机
    """

    def get_status(self) -> AgentFSMStateEnum:
        return AgentFSMStateEnum.AFTER_LLM_OUTPUT

    async def execute(self, fsm: "AgentFSM") -> AsyncGenerator[BaseEvent, None]:
        sd = fsm.shared_data
        llm_output = sd.llm_output

        # 将 LLM 输出存入 raw_messages
        sd.context.add_raw_message(llm_output)

        # 调用 after_llm_output 生命周期钩子
        handler = sd.lifespan_manager.get_lifespan("after_llm_output")
        request = AfterLLMOutputRequest(
            llm_config=sd.llm_config,
            context=sd.context,
            event_channel=sd.event_channel,
            llm_output=llm_output,
            kwargs=sd.lifespan_manager.get_kwargs(),
        )
        response: AfterLLMOutputResponse = await handler.execute(request)

        # 回写
        sd.lifespan_manager.set_kwargs(response.kwargs)
        sd.context = response.context
        sd.llm_config = response.llm_config
        sd.llm_output = response.llm_output

        # 将处理后的 LLM 输出存入 work_messages
        sd.context.add_work_message(sd.llm_output)

        # ===== 路由决策 =====
        finish_reason = sd.llm_output.finish_reason

        if finish_reason == "tool_calls":
            # 路由到工具执行流水线
            fsm.next_state = BeforeExecuteToolsState()

        elif finish_reason == "stop":
            # 正常停止，进入结束处理
            fsm.next_state = AfterFinishState()

        else:
            # 其他停止原因（length / content_filter / balance / error 等），直接结束
            yield RoundStopEvent(
                data=RoundStopEvent.RoundStopEventData(
                    finish_reason=finish_reason,
                    llm_config=sd.llm_config,
                    context=sd.context,
                    kwargs=sd.lifespan_manager.get_kwargs(),
                )
            )
            fsm.next_state = None


# ======= 工具执行前处理状态 =======

class BeforeExecuteToolsState(BaseAgentFSMState):
    """
    工具执行前处理状态

    执行逻辑：
    1. 调用 before_execute_tools 生命周期钩子
       - 默认实现中包含 Guard 检测逻辑
       - 若检测到 Guard 触发，会抛出 ToolCallGuardTriggeredException
       - 该异常由 AgentFSM 主循环捕获处理
    2. 将 Guard 过滤后的 finished_tool_calls 写入共享数据
    3. 流转到 ExecutingToolsState
    """

    def get_status(self) -> AgentFSMStateEnum:
        return AgentFSMStateEnum.BEFORE_EXECUTE_TOOLS

    async def execute(self, fsm: "AgentFSM") -> AsyncGenerator[BaseEvent, None]:
        sd = fsm.shared_data

        # 构建请求数据
        request = BeforeExecuteToolsRequest(
            llm_config=sd.llm_config,
            context=sd.context,
            event_channel=sd.event_channel,
            llm_output=sd.llm_output,
            kwargs=sd.lifespan_manager.get_kwargs(),
        )

        # 如果携带了 human_response（恢复流程），注入到请求中
        if sd.human_response is not None:
            request.human_response = sd.human_response

        # 调用 before_execute_tools 生命周期钩子
        # 注意：默认实现中的 Guard 检测可能抛出 ToolCallGuardTriggeredException
        handler = sd.lifespan_manager.get_lifespan("before_execute_tools")
        response: BeforeExecuteToolsResponse = await handler.execute(request)

        # 回写
        sd.lifespan_manager.set_kwargs(response.kwargs)
        sd.context = response.context
        sd.llm_config = response.llm_config
        sd.llm_output = response.llm_output
        sd.pending_tool_calls = response.pending_tool_calls
        sd.finished_tool_calls = response.finished_tool_calls
        sd.prebuilt_tool_results = response.prebuilt_tool_results

        # 清除 human_response（一次性使用）
        sd.human_response = None

        # 流转到工具执行状态
        fsm.next_state = ExecutingToolsState()
        return
        yield  # 使函数成为异步生成器（本状态无事件产出）


# ======= 工具执行状态 =======

class ExecutingToolsState(BaseAgentFSMState):
    """
    工具执行状态

    执行逻辑：
    1. 调用 executing_tools 生命周期钩子（默认实现会执行实际的工具调用）
    2. 产出 ToolsExecutedEvent 事件
    3. 流转到 AfterExecuteToolsState
    """

    def get_status(self) -> AgentFSMStateEnum:
        return AgentFSMStateEnum.EXECUTING_TOOLS

    async def execute(self, fsm: "AgentFSM") -> AsyncGenerator[BaseEvent, None]:
        sd = fsm.shared_data

        # 调用 executing_tools 生命周期钩子
        handler = sd.lifespan_manager.get_lifespan("executing_tools")
        request = ExecutingToolsRequest(
            llm_config=sd.llm_config,
            context=sd.context,
            event_channel=sd.event_channel,
            llm_output=sd.llm_output,
            user_input=sd.user_input,
            kwargs=sd.lifespan_manager.get_kwargs(),
            pending_tool_calls=sd.pending_tool_calls or [],
            finished_tool_calls=sd.finished_tool_calls or [],
            prebuilt_tool_results=sd.prebuilt_tool_results or [],
        )
        response: ExecutingToolsResponse = await handler.execute(request)

        # 回写
        sd.lifespan_manager.set_kwargs(response.kwargs)
        sd.context = response.context
        sd.llm_config = response.llm_config
        sd.llm_output = response.llm_output
        sd.finished_tool_calls = response.finished_tool_calls
        sd.tool_results = response.tool_results
        sd.pending_tool_calls = None
        sd.prebuilt_tool_results = None

        # 产出工具执行完毕事件
        yield ToolsExecutedEvent(
            data=ToolsExecutedEvent.ToolsExecutedEventData(
                tool_results=sd.tool_results,
            )
        )

        # 流转到工具执行后处理状态
        fsm.next_state = AfterExecuteToolsState()


# ======= 工具执行后处理状态 =======

class AfterExecuteToolsState(BaseAgentFSMState):
    """
    工具执行后处理状态

    执行逻辑：
    1. 将工具执行结果添加到消息列表（raw_messages + work_messages）
    2. 调用 after_execute_tools 生命周期钩子
    3. 流转回 LLMOutputState，继续 Agent 主循环
    """

    def get_status(self) -> AgentFSMStateEnum:
        return AgentFSMStateEnum.AFTER_EXECUTE_TOOLS

    async def execute(self, fsm: "AgentFSM") -> AsyncGenerator[BaseEvent, None]:
        sd = fsm.shared_data
        tool_results: List = sd.tool_results or []

        # 将工具结果存入 raw_messages
        sd.context.add_raw_messages(tool_results)

        # 调用 after_execute_tools 生命周期钩子
        handler = sd.lifespan_manager.get_lifespan("after_execute_tools")
        request = AfterExecuteToolsRequest(
            llm_config=sd.llm_config,
            context=sd.context,
            event_channel=sd.event_channel,
            llm_output=sd.llm_output,
            tool_results=tool_results,
            kwargs=sd.lifespan_manager.get_kwargs(),
        )
        response: AfterExecuteToolsResponse = await handler.execute(request)

        # 回写
        sd.lifespan_manager.set_kwargs(response.kwargs)
        sd.context = response.context
        sd.llm_config = response.llm_config
        sd.llm_output = response.llm_output
        sd.tool_results = response.tool_results

        # 将工具结果存入 work_messages
        sd.context.add_work_messages(sd.tool_results)

        # 回到 LLM 输出状态，继续 Agent 主循环
        fsm.next_state = LLMOutputState()
        return
        yield  # 使函数成为异步生成器（本状态无事件产出）


# ======= 轮次结束处理状态 =======

class AfterFinishState(BaseAgentFSMState):
    """
    轮次结束处理状态

    执行逻辑：
    1. 调用 after_finish 生命周期钩子
    2. 产出 RoundStopEvent 事件
    3. 设置 next_state = None 终止状态机
    """

    def get_status(self) -> AgentFSMStateEnum:
        return AgentFSMStateEnum.AFTER_FINISH

    async def execute(self, fsm: "AgentFSM") -> AsyncGenerator[BaseEvent, None]:
        sd = fsm.shared_data

        # 调用 after_finish 生命周期钩子
        handler = sd.lifespan_manager.get_lifespan("after_finish")
        request = AfterFinishRequest(
            llm_config=sd.llm_config,
            context=sd.context,
            event_channel=sd.event_channel,
            kwargs=sd.lifespan_manager.get_kwargs(),
        )
        response: AfterFinishResponse = await handler.execute(request)

        # 回写
        sd.lifespan_manager.set_kwargs(response.kwargs)
        sd.context = response.context
        sd.llm_config = response.llm_config

        # 产出轮次结束事件
        yield RoundStopEvent(
            data=RoundStopEvent.RoundStopEventData(
                finish_reason="stop",
                llm_config=sd.llm_config,
                context=sd.context,
                kwargs=sd.lifespan_manager.get_kwargs(),
            )
        )

        # 终止状态机
        fsm.next_state = None
