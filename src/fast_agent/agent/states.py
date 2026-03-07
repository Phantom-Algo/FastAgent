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

from typing import AsyncGenerator

from .fsm import IAgentState, AgentFSM
from .state import AgentState
from .event import (
    BaseEvent,
    RoundStopEvent,
    ToolsExecutedEvent,
)
from .lifespan import (
    InputAfterUserInput,
    OutputAfterUserInput,
    InputAfterLLMOutput,
    OutputAfterLLMOutput,
    InputBeforeExecuteTools,
    OutputBeforeExecuteTools,
    InputExecutingTools,
    OutputExecutingTools,
    InputAfterExecuteTools,
    OutputAfterExecuteTools,
    InputAfterFinish,
    OutputAfterFinish,
)
from .adapter import AdapterFactory
from fast_agent.llm import AssistantMessage


# ======= 用户输入处理状态 =======

class AfterUserInputState(IAgentState):
    """
    用户输入处理状态

    执行逻辑：
    1. 将用户输入添加到原始消息列表（raw_messages）
    2. 调用 after_user_input 生命周期钩子
    3. 将处理后的用户输入添加到工作消息列表（work_messages）
    4. 流转到 LLMOutputState
    """

    def get_status(self) -> AgentState:
        return AgentState.AFTER_USER_INPUT

    async def execute(self, fsm: AgentFSM) -> AsyncGenerator[BaseEvent, None]:
        agent = fsm.agent
        user_input = fsm.user_input

        # 将用户原始输入存入 raw_messages
        agent.context.add_raw_message(user_input)

        # 调用 after_user_input 生命周期钩子
        input_data = InputAfterUserInput.from_agent(agent, user_input=user_input)
        output_data: OutputAfterUserInput = await agent.lifespan.after_user_input.after_user_input(input_data)
        output_data.update_agent(agent)

        # 更新 FSM 共享数据，将处理后的用户输入存入 work_messages
        fsm.user_input = output_data.user_input
        agent.context.add_work_message(fsm.user_input)

        # 流转到 LLM 输出状态
        fsm.next_state = LLMOutputState()
        return
        yield  # 使函数成为异步生成器（本状态无事件产出）


# ======= LLM 流式输出状态 =======

class LLMOutputState(IAgentState):
    """
    LLM 流式输出状态

    执行逻辑：
    1. 根据 llm_config 获取对应的 Adapter
    2. 调用 Adapter.stream() 进行流式输出
    3. 逐个产出 chunk 事件和 tool_call 事件
    4. 接收到完整的 AssistantMessage 后流转到 AfterLLMOutputState
    """

    def get_status(self) -> AgentState:
        return AgentState.LLM_OUTPUT

    async def execute(self, fsm: AgentFSM) -> AsyncGenerator[BaseEvent, None]:
        agent = fsm.agent

        # 根据 llm_config 获取 Adapter
        adapter_cls = AdapterFactory.get_adapter_cls(provider=agent.llm_config.provider)
        adapter = adapter_cls()

        # 调用 Adapter 流式输出，逐个产出事件
        llm_output = None
        async for output in adapter.stream(
            llm_config=agent.llm_config,
            context=agent.context,
        ):
            event = agent._wrap_to_event(output)
            yield event

            # 如果收到完整的 AssistantMessage，说明流式输出结束
            if isinstance(output, AssistantMessage):
                llm_output = output
                break

        if llm_output is None:
            raise ValueError("LLM 流式输出异常终止：未接收到 AssistantMessage。")

        # 更新 FSM 共享数据，流转到 after_llm_output 状态
        fsm.llm_output = llm_output
        fsm.next_state = AfterLLMOutputState()


# ======= LLM 输出后处理与路由状态 =======

class AfterLLMOutputState(IAgentState):
    """
    LLM 输出后处理与路由状态

    执行逻辑：
    1. 将 LLM 输出添加到消息列表（raw_messages + work_messages）
    2. 调用 after_llm_output 生命周期钩子
    3. 根据 finish_reason 进行路由决策：
       - "tool_calls" → BeforeExecuteToolsState（进入工具执行流水线）
       - "stop"       → AfterFinishState（正常结束流程）
       - 其他         → 产出 RoundStopEvent 并终止状态机
    """

    def get_status(self) -> AgentState:
        return AgentState.AFTER_LLM_OUTPUT

    async def execute(self, fsm: AgentFSM) -> AsyncGenerator[BaseEvent, None]:
        agent = fsm.agent
        llm_output = fsm.llm_output

        # 将 LLM 输出存入 raw_messages
        agent.context.add_raw_message(llm_output)

        # 调用 after_llm_output 生命周期钩子
        input_data = InputAfterLLMOutput.from_agent(agent, llm_output=llm_output)
        output_data: OutputAfterLLMOutput = await agent.lifespan.after_llm_output.after_llm_output(input_data)
        output_data.update_agent(agent)

        # 更新 FSM 共享数据
        fsm.llm_output = output_data.llm_output
        agent.context.add_work_message(fsm.llm_output)

        # ===== 路由决策 =====
        finish_reason = fsm.llm_output.finish_reason

        if finish_reason == "tool_calls":
            # 路由到工具执行流水线
            fsm.next_state = BeforeExecuteToolsState()

        elif finish_reason == "stop":
            # 正常停止，进入结束处理
            fsm.next_state = AfterFinishState()

        else:
            # 其他停止原因（length / content_filter / balance / error 等），直接结束
            # TODO: 后续可在此处增加生命周期钩子，让开发者自定义不同停止原因的处理逻辑
            yield RoundStopEvent(
                data=RoundStopEvent.RoundStopEventData(
                    finish_reason=finish_reason,
                    llm_config=agent.llm_config,
                    context=agent.context,
                    kwargs=agent.lifespan.kwargs,
                )
            )
            fsm.next_state = None


# ======= 工具执行前处理状态 =======

class BeforeExecuteToolsState(IAgentState):
    """
    工具执行前处理状态

    执行逻辑：
    1. 调用 before_execute_tools 生命周期钩子（开发者可在此修改/过滤工具调用）
    2. 流转到 ExecutingToolsState
    """

    def get_status(self) -> AgentState:
        return AgentState.BEFORE_EXECUTE_TOOLS

    async def execute(self, fsm: AgentFSM) -> AsyncGenerator[BaseEvent, None]:
        agent = fsm.agent

        # 调用 before_execute_tools 生命周期钩子
        input_data = InputBeforeExecuteTools.from_agent(agent, llm_output=fsm.llm_output)
        output_data: OutputBeforeExecuteTools = await agent.lifespan.before_execute_tools.before_execute_tools(input_data)
        output_data.update_agent(agent)
        fsm.llm_output = output_data.llm_output

        # 流转到工具执行状态
        fsm.next_state = ExecutingToolsState()
        return
        yield  # 使函数成为异步生成器（本状态无事件产出）


# ======= 工具执行状态 =======

class ExecutingToolsState(IAgentState):
    """
    工具执行状态

    执行逻辑：
    1. 调用 executing_tools 生命周期钩子（默认实现会执行实际的工具调用）
    2. 产出 ToolsExecutedEvent 事件
    3. 流转到 AfterExecuteToolsState
    """

    def get_status(self) -> AgentState:
        return AgentState.EXECUTING_TOOLS

    async def execute(self, fsm: AgentFSM) -> AsyncGenerator[BaseEvent, None]:
        agent = fsm.agent

        # 调用 executing_tools 生命周期钩子
        input_data = InputExecutingTools.from_agent(
            agent, 
            llm_output=fsm.llm_output,
            human_response=fsm.human_review_response,
            finished_tool_calls=fsm.finished_tool_calls,
        )
        output_data: OutputExecutingTools = await agent.lifespan.executing_tools.executing_tools(input_data)
        output_data.update_agent(agent)

        # 更新 FSM 共享数据
        fsm.llm_output = output_data.llm_output
        fsm.tool_results = output_data.tool_results

        # 产出工具执行完毕事件
        yield ToolsExecutedEvent(
            data=ToolsExecutedEvent.ToolsExecutedEventData(
                tool_results=fsm.tool_results,
            )
        )

        # 流转到工具执行后处理状态
        fsm.next_state = AfterExecuteToolsState()


# ======= 工具执行后处理状态 =======

class AfterExecuteToolsState(IAgentState):
    """
    工具执行后处理状态

    执行逻辑：
    1. 将工具执行结果添加到消息列表（raw_messages + work_messages）
    2. 调用 after_execute_tools 生命周期钩子
    3. 流转回 LLMOutputState，继续 Agent 主循环
    """

    def get_status(self) -> AgentState:
        return AgentState.AFTER_EXECUTE_TOOLS

    async def execute(self, fsm: AgentFSM) -> AsyncGenerator[BaseEvent, None]:
        agent = fsm.agent

        # 将工具结果存入 raw_messages
        agent.context.add_raw_messages(fsm.tool_results)

        # 调用 after_execute_tools 生命周期钩子
        input_data = InputAfterExecuteTools.from_agent(
            agent, llm_output=fsm.llm_output, tool_results=fsm.tool_results,
        )
        output_data: OutputAfterExecuteTools = await agent.lifespan.after_execute_tools.after_execute_tools(input_data)
        output_data.update_agent(agent)

        # 更新 FSM 共享数据
        fsm.llm_output = output_data.llm_output
        fsm.tool_results = output_data.tool_results

        # 将工具结果存入 work_messages
        agent.context.work_messages.add_messages(fsm.tool_results)

        # 回到 LLM 输出状态，继续 Agent 主循环
        fsm.next_state = LLMOutputState()
        return
        yield  # 使函数成为异步生成器（本状态无事件产出）


# ======= 正常结束处理状态 =======

class AfterFinishState(IAgentState):
    """
    正常结束处理状态

    执行逻辑：
    1. 调用 after_finish 生命周期钩子
    2. 产出 RoundStopEvent（finish_reason="stop"）
    3. 终止状态机
    """

    def get_status(self) -> AgentState:
        return AgentState.AFTER_FINISH

    async def execute(self, fsm: AgentFSM) -> AsyncGenerator[BaseEvent, None]:
        agent = fsm.agent

        # 调用 after_finish 生命周期钩子
        input_data = InputAfterFinish.from_agent(agent)
        output_data: OutputAfterFinish = await agent.lifespan.after_finish.after_finish(input_data)
        output_data.update_agent(agent)

        # 产出轮次正常结束事件
        yield RoundStopEvent(
            data=RoundStopEvent.RoundStopEventData(
                finish_reason="stop",
                llm_config=agent.llm_config,
                context=agent.context,
                kwargs=agent.lifespan.kwargs,
            )
        )

        # 终止状态机
        fsm.next_state = None
