"""
默认生命周期实现模块

提供 Agent 6 个生命周期阶段的默认实现：
- DefaultAfterUserInput: 用户输入后，透传
- DefaultAfterLLMOutput: LLM 输出后，透传
- DefaultBeforeExecuteTools: 工具执行前，⭐ Guard 检测核心逻辑
- DefaultExecutingTools: 工具执行中，⭐ 工具执行引擎
- DefaultAfterExecuteTools: 工具执行后，透传
- DefaultAfterFinish: 轮次结束后，透传

Guard 机制说明：
    在 BeforeExecuteTools 阶段检测带 GuardPolicy 的工具调用，
    若未携带 human_response 则抛出 ToolCallGuardTriggeredException 中断 Agent。
    恢复时携带 human_response，通过 guard_func 验证后继续执行。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List

from ....types.agent.lifespan.base_lifespan import (
    IAfterExecuteTools,
    IAfterFinish,
    IAfterLLMOutput,
    IAfterUserInput,
    IBeforeExecuteTools,
    IExecutingTools,
)
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
from ....types.messages.domain.assistant_message import ToolCall
from ....types.messages.domain.tool_result_message import ToolResultMessage
from ....types.tool.base_tool import BaseTool
from ....types.tool.domain.guard_triggered import (
    ToolCallGuardTriggeredContext,
    ToolCallGuardTriggeredException,
)
from ....types.agent.exceptions.tool_execution_interrupted_exception import ToolExecutionInterruptedException

logger = logging.getLogger(__name__)


# ================================================================
# 透传类生命周期（直接返回输入数据，不做额外处理）
# ================================================================

class DefaultAfterUserInput(IAfterUserInput):
    """默认用户输入后处理：直接透传，不做额外处理。"""

    async def execute(self, data: AfterUserInputRequest) -> AfterUserInputResponse:
        return AfterUserInputResponse(
            llm_config=data.llm_config,
            context=data.context,
            event_channel=data.event_channel,
            user_input=data.user_input,
            kwargs=data.kwargs,
        )


class DefaultAfterLLMOutput(IAfterLLMOutput):
    """默认 LLM 输出后处理：直接透传，不做额外处理。"""

    async def execute(self, data: AfterLLMOutputRequest) -> AfterLLMOutputResponse:
        return AfterLLMOutputResponse(
            llm_config=data.llm_config,
            context=data.context,
            event_channel=data.event_channel,
            llm_output=data.llm_output,
            user_input=data.user_input,
            kwargs=data.kwargs,
        )


class DefaultAfterExecuteTools(IAfterExecuteTools):
    """默认工具执行后处理：直接透传，不做额外处理。"""

    async def execute(self, data: AfterExecuteToolsRequest) -> AfterExecuteToolsResponse:
        return AfterExecuteToolsResponse(
            llm_config=data.llm_config,
            context=data.context,
            event_channel=data.event_channel,
            llm_output=data.llm_output,
            tool_results=data.tool_results,
            kwargs=data.kwargs,
            user_input=data.user_input,
        )


class DefaultAfterFinish(IAfterFinish):
    """默认轮次结束处理：直接透传，不做额外处理。"""

    async def execute(self, data: AfterFinishRequest) -> AfterFinishResponse:
        return AfterFinishResponse(
            llm_config=data.llm_config,
            context=data.context,
            event_channel=data.event_channel,
            kwargs=data.kwargs,
            user_input=data.user_input,
            llm_output=data.llm_output,
        )


# ================================================================
# Guard 检测核心 —— BeforeExecuteTools
# ================================================================

class DefaultBeforeExecuteTools(IBeforeExecuteTools):
    """
    默认工具执行前处理：Guard 检测核心逻辑。
    """

    async def execute(self, data: BeforeExecuteToolsRequest) -> BeforeExecuteToolsResponse:
        llm_output = data.llm_output
        tool_calls: List[ToolCall] = llm_output.tool_calls if llm_output and llm_output.tool_calls else []

        if not tool_calls:
            return BeforeExecuteToolsResponse(
                user_input=data.user_input,
                llm_config=data.llm_config,
                context=data.context,
                event_channel=data.event_channel,
                llm_output=data.llm_output,
                kwargs=data.kwargs,
                pending_tool_calls=[],
                finished_tool_calls=[],
                prebuilt_tool_results=[],
            )

        # 获取所有已注册的工具，按名称索引
        tools = data.context.get_tool_manager().get_tools()
        tools_by_name: Dict[str, BaseTool] = {t.name: t for t in tools}

        # 获取 human_response（恢复流程中携带）
        human_response = data.human_response

        # 分类列表
        pending_tool_calls: List[ToolCall] = []  # 通过 guard / 无 guard，待执行的 tool_call
        finished_tool_calls: List[ToolCall] = []  # 在 Before 阶段已完成（如 Guard 拒绝）的 tool_call
        guard_triggered_contexts: List[ToolCallGuardTriggeredContext] = []  # 待人工审批的 guard 上下文
        reject_tool_results: List[ToolResultMessage] = []  # guard 拒绝后生成的 ToolResultMessage

        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.function_name)

            # 工具不存在或无 guard_policy → 直接放行
            if tool is None or tool.guard_policy is None:
                pending_tool_calls.append(tool_call)
                continue

            guard_policy = tool.guard_policy

            # 检查是否存在对应的 human_response（根据 tool_call_id 进行检索）
            call_response = human_response.get(tool_call.tool_call_id) if human_response else None

            if call_response is None:
                # 无 human_response → Guard 触发，收集上下文
                guard_triggered_contexts.append(
                    ToolCallGuardTriggeredContext(
                        tool_call=tool_call,
                        tool_info=tool,
                    )
                )
            else:
                # 有 human_response → 执行 guard_func 验证
                if guard_policy.guard_func is not None:
                    try:
                        passed = guard_policy.guard_func(call_response)
                    except Exception as e:
                        passed = False
                else:
                    # 无 guard_func，默认拒绝
                    passed = False

                if passed:
                    pending_tool_calls.append(tool_call)
                else:
                    # Guard 拒绝 → 调用 reject_func 生成拒绝结果
                    if guard_policy.reject_func is not None:
                        try:
                            reject_result = guard_policy.reject_func(call_response)
                            # 确保 tool_call_id 和 name 与实际工具调用匹配
                            reject_result.tool_call_id = tool_call.tool_call_id
                            reject_result.name = tool_call.function_name
                        except Exception as e:
                            reject_result = ToolResultMessage(
                                tool_call_id=tool_call.tool_call_id,
                                name=tool_call.function_name,
                                content=f"Tool `{tool_call.function_name}` was rejected by guard policy.",
                                is_error=False,
                            )
                    else:
                        reject_result = ToolResultMessage(
                            tool_call_id=tool_call.tool_call_id,
                            name=tool_call.function_name,
                            content=f"Tool `{tool_call.function_name}` was rejected by guard policy.",
                            is_error=False,
                        )
                    reject_tool_results.append(reject_result)
                    finished_tool_calls.append(tool_call)

        # 存在需要人工审批的 guard，抛出异常中断 Agent
        if guard_triggered_contexts:
            raise ToolCallGuardTriggeredException(
                message=f"Guard triggered for {len(guard_triggered_contexts)} tool call(s), awaiting human response.",
                contexts=guard_triggered_contexts
            )

        return BeforeExecuteToolsResponse(
            llm_config=data.llm_config,
            context=data.context,
            event_channel=data.event_channel,
            llm_output=data.llm_output,
            user_input=data.user_input,
            kwargs=data.kwargs,
            pending_tool_calls=pending_tool_calls,
            finished_tool_calls=finished_tool_calls,
            prebuilt_tool_results=reject_tool_results,
        )


# ================================================================
# 工具执行引擎 —— ExecutingTools
# ================================================================

class DefaultExecutingTools(IExecutingTools):
    """
    默认工具执行引擎。
    """

    async def execute(self, data: ExecutingToolsRequest) -> ExecutingToolsResponse:
        llm_output = data.llm_output
        pending_tool_calls: List[ToolCall] = data.pending_tool_calls or []
        finished_tool_calls: List[ToolCall] = data.finished_tool_calls or []
        prebuilt_tool_results: List[ToolResultMessage] = data.prebuilt_tool_results or []

        if not pending_tool_calls:
            # 无需执行的工具调用，直接返回预封装结果（如 Guard 拒绝结果）
            return ExecutingToolsResponse(
                llm_config=data.llm_config,
                context=data.context,
                event_channel=data.event_channel,
                llm_output=llm_output,
                finished_tool_calls=finished_tool_calls,
                tool_results=prebuilt_tool_results,
                kwargs=data.kwargs,
            )

        # 获取工具列表并按名称索引
        tools = data.context.get_tool_manager().get_tools()
        tools_by_name: Dict[str, BaseTool] = {t.name: t for t in tools}

        # 获取工具参数注入值
        tool_inject_params = data.context.get_tool_inject_params()

        async def _run_single_tool_call(tool_call: ToolCall) -> ToolResultMessage:
            """执行单个工具调用，包含参数注入和错误处理。"""
            tool = tools_by_name.get(tool_call.function_name)

            if tool is None:
                return ToolResultMessage(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.function_name,
                    content=f"Tool `{tool_call.function_name}` not found or not registered.",
                    is_error=False,
                )

            # 构建调用参数
            call_kwargs = dict(tool_call.function_args or {})

            # 注入参数（inject_params 中声明的参数从 context 获取）
            for param_name in (tool.inject_params or []):
                if param_name in tool_inject_params:
                    call_kwargs[param_name] = tool_inject_params[param_name]

            # 构建 ToolRuntime 并注入（如果工具声明了 tool_runtime_param_name）
            if tool.tool_runtime_param_name:
                from ...tool.tool_runtime import ToolRuntime
                from ...tool.ask_human_channel import AskHumanChannel

                ask_human_channel = AskHumanChannel(event_channel=data.event_channel)
                tool_runtime = ToolRuntime(
                    tool_call_id=tool_call.tool_call_id,
                    this_tool=tool,
                    llm_config=data.llm_config,
                    context=data.context,
                    llm_output=llm_output,
                    user_input=data.user_input,
                    ask_human_channel=ask_human_channel,
                    kwars=data.kwargs,
                )
                call_kwargs[tool.tool_runtime_param_name] = tool_runtime

            try:
                if tool.is_async:
                    result = await tool(**call_kwargs)
                else:
                    result = await asyncio.to_thread(tool, **call_kwargs)

                return ToolResultMessage(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool.name,
                    content=result,
                    is_error=False,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Tool `{tool.name}` execution failed (call_id={tool_call.tool_call_id})."
                ) from e

        # 并发执行所有工具调用，并保留异常以便构建可恢复的中断快照。
        execution_results = await asyncio.gather(
            *[_run_single_tool_call(tc) for tc in pending_tool_calls],
            return_exceptions=True,
        )

        succeeded_tool_calls: List[ToolCall] = []
        succeeded_tool_results: List[ToolResultMessage] = []
        failed_tool_calls: List[ToolCall] = []

        for tool_call, execution_result in zip(pending_tool_calls, execution_results):
            if isinstance(execution_result, Exception):
                failed_tool_calls.append(tool_call)
                
            else:
                succeeded_tool_calls.append(tool_call)
                succeeded_tool_results.append(execution_result)

        updated_finished_tool_calls = finished_tool_calls + succeeded_tool_calls

        # 合并预封装结果（排在已执行结果之前）
        all_results = prebuilt_tool_results + succeeded_tool_results

        if failed_tool_calls:
            raise ToolExecutionInterruptedException(
                message=f"{len(failed_tool_calls)} tool call(s) failed during execution.",
                pending_tool_calls=failed_tool_calls,
                finished_tool_calls=updated_finished_tool_calls,
                tool_results=all_results,
            )

        return ExecutingToolsResponse(
            llm_config=data.llm_config,
            context=data.context,
            event_channel=data.event_channel,
            llm_output=llm_output,
            finished_tool_calls=updated_finished_tool_calls,
            tool_results=all_results,
            kwargs=data.kwargs,
        )
