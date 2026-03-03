from fast_agent.llm import (
    LLMConfig, 
    Context, 
    UserMessage, 
    AssistantMessageChunk,
    ToolCall,
    AssistantMessage,
    ToolResultMessage
)
from .lifespan import (
    Lifespan,
    IAfterUserInput,
    IAfterLLMOutput,
    IBeforeExecuteTools,
    IExecutingTools,
    IAfterExecuteTools,
    IAfterFinish,
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
    OutputAfterFinish
)
from .adapter import AdapterFactory
from .event import (
    BaseEvent, 
    AssistantMessageChunkOutputEvent,
    ToolCallEvent,
    AssistantMessageOutputEvent,
    RoundStopEvent,
    ToolsExecutedEvent
)
from typing import Optional, AsyncGenerator, Any, Dict
import asyncio
from copy import deepcopy


# ===== 生命周期默认实现 =====
class AfterUserInput(IAfterUserInput):
    async def after_user_input(self, data):
        return OutputAfterUserInput(
            llm_config=data.llm_config,
            context=data.context,
            user_input=data.user_input,
            kwargs=data.kwargs,
        )
    
class AfterLLMOutput(IAfterLLMOutput):
    async def after_llm_output(self, data):
        return OutputAfterLLMOutput(
            llm_config=data.llm_config,
            context=data.context,
            llm_output=data.llm_output,
            kwargs=data.kwargs,
        )
    
class BeforeExecuteTools(IBeforeExecuteTools):
    async def before_execute_tools(self, data):
        return OutputBeforeExecuteTools(
            llm_config=data.llm_config,
            context=data.context,
            llm_output=data.llm_output,
            kwargs=data.kwargs,
        )
    
class ExecutingTools(IExecutingTools):
    async def executing_tools(self, data):
        llm_output = data.llm_output
        tool_calls = llm_output.tool_calls or []

        if not tool_calls:
            return OutputExecutingTools(
                llm_config=data.llm_config,
                context=data.context,
                llm_output=llm_output,
                tool_results=[],
                kwargs=data.kwargs
            )

        tools = data.context.tools.get_tools()
        tools_by_name = {tool.name: tool for tool in tools}
        tool_inject_params = data.context.get_tool_inject_params()

        async def _run_single_tool_call(tool_call: ToolCall) -> ToolResultMessage:
            tool = tools_by_name.get(tool_call.function_name)
            if tool is None:
                return ToolResultMessage(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.function_name,
                    content=f"工具 `{tool_call.function_name}` 调用失败：工具不存在或未注册。",
                    is_error=False
                )

            call_kwargs = dict(tool_call.function_args or {})
            for param_name in (tool.inject_params or []):
                if param_name in tool_inject_params:
                    call_kwargs[param_name] = tool_inject_params[param_name]

            try:
                if tool.is_async:
                    result = await tool(**call_kwargs)
                else:
                    result = await asyncio.to_thread(tool, **call_kwargs)

                return ToolResultMessage(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool.name,
                    content=result,
                    is_error=False
                )
            except Exception:
                # 脱敏：不向 LLM 暴露具体错误信息，防止环境信息泄露
                return ToolResultMessage(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool.name,
                    content=f"工具 `{tool.name}` 调用失败，请基于已有信息继续回答。",
                    is_error=False
                )

        # gather 返回结果顺序与传入顺序一致，可同时执行异步/同步工具调用
        tool_results = await asyncio.gather(*[_run_single_tool_call(tool_call) for tool_call in tool_calls])

        return OutputExecutingTools(
            llm_config=data.llm_config,
            context=data.context,
            llm_output=llm_output,
            tool_results=tool_results,
            kwargs=data.kwargs
        )
    
class AfterExecuteTools(IAfterExecuteTools):
    async def after_execute_tools(self, data):
        return OutputAfterExecuteTools(
            llm_config=data.llm_config,
            context=data.context,
            llm_output=data.llm_output,
            tool_results=data.tool_results,
            kwargs=data.kwargs,
        )
    
class AfterFinish(IAfterFinish):
    async def after_finish(self, data):
        return OutputAfterFinish(
            llm_config=data.llm_config,
            context=data.context,
            kwargs=data.kwargs,
        )


# ===== Agent 核心类 =====
class Agent:

    def __init__(
        self,
        llm_config: LLMConfig,
        context: Context,
        lifespan: Optional[Lifespan] = None
    ):
        self.llm_config = llm_config
        self.context = context
        self.lifespan = self._normalize_lifespan(lifespan=lifespan)

    async def stream(self, user_input: UserMessage) -> AsyncGenerator[BaseEvent, None]:
        # ==== 第一阶段：after_user_input ====
        raw_user_input = deepcopy(user_input)
        self.context.raw_messages.add_message(raw_user_input)
        self.context.subsequent_messages.append(raw_user_input)

        input_after_user_input = InputAfterUserInput(
            self.llm_config,
            self.context,
            user_input=user_input,
            kwargs=self.lifespan.kwargs
        )
        output_after_user_input: OutputAfterUserInput = await self.lifespan.after_user_input.after_user_input(input_after_user_input)

        self.llm_config = output_after_user_input.llm_config
        self.context = output_after_user_input.context
        self.lifespan.kwargs = output_after_user_input.kwargs
        user_input = output_after_user_input.user_input

        self.context.work_messages.add_message(user_input)

        # 根据 llm_config 获取 Adapter
        adapter_cls = AdapterFactory.get_adapter_cls(provider=self.llm_config.provider)
        adapter = adapter_cls()

        # Agent 主循环
        while(True):
            # 将 llm_config 和 context 传入 Adapter，获取流式输出内容以及最后包装好的 llm_output
            llm_output = None
            async for output in adapter.stream(
                llm_config=self.llm_config,
                context=self.context
            ):
                event = self._wrap_to_event(output)
                yield event

                # 如果是 AssistantMessage，说明流式输出结束，进入第二阶段 after_llm_output
                if isinstance(output, AssistantMessage):
                    llm_output = output
                    break

            if llm_output is None:
                raise ValueError("LLM output is None after streaming, expected an AssistantMessage.")
            


            # ==== 第二阶段：after_llm_output ====
            raw_llm_output = deepcopy(llm_output)
            self.context.raw_messages.add_message(raw_llm_output)
            self.context.subsequent_messages.append(raw_llm_output)

            input_after_llm_output = InputAfterLLMOutput(
                llm_config=self.llm_config,
                context=self.context,
                llm_output=llm_output,
                kwargs=self.lifespan.kwargs
            )
            output_after_llm_output: OutputAfterLLMOutput = await self.lifespan.after_llm_output.after_llm_output(input_after_llm_output)
            
            self.llm_config = output_after_llm_output.llm_config
            self.context = output_after_llm_output.context
            self.lifespan.kwargs = output_after_llm_output.kwargs
            llm_output = output_after_llm_output.llm_output

            self.context.work_messages.add_message(llm_output)

            # 路由
            if llm_output.finish_reason != "tool_calls":
                # 如果不是工具调用结束，则说明 Agent 任务结束
                if llm_output.finish_reason == "stop":
                    # 如果是正常停止，进入最后阶段 after_finish
                    input_after_finish = InputAfterFinish(
                        llm_config=self.llm_config,
                        context=self.context,
                        kwargs=self.lifespan.kwargs
                    )
                    output_after_finish: OutputAfterFinish = await self.lifespan.after_finish.after_finish(input_after_finish)

                    self.llm_config = output_after_finish.llm_config
                    self.context = output_after_finish.context
                    self.lifespan.kwargs = output_after_finish.kwargs

                    yield RoundStopEvent(
                        data=RoundStopEvent.RoundStopEventData(
                            finish_reason="stop",
                            llm_config=self.llm_config,
                            context=self.context,
                            kwargs=self.lifespan.kwargs
                        )
                    )
                
                else:
                    # 其它原因停止
                    # TODO:后续可以在这里加一个生命周期，让用户根据不同的停止原因进行不同的处理
                    yield RoundStopEvent(
                        data=RoundStopEvent.RoundStopEventData(
                            finish_reason=llm_output.finish_reason,
                            llm_config=self.llm_config,
                            context=self.context,
                            kwargs=self.lifespan.kwargs
                        )
                    )
                
                # 无论如何都结束当前 Agent 循环
                break

            # ==== 第三阶段：before_execute_tools ====
            input_before_execute_tools = InputBeforeExecuteTools(
                llm_config=self.llm_config,
                context=self.context,
                llm_output=llm_output,
                kwargs=self.lifespan.kwargs
            )
            output_before_execute_tools: OutputBeforeExecuteTools = await self.lifespan.before_execute_tools.before_execute_tools(input_before_execute_tools)

            self.llm_config = output_before_execute_tools.llm_config
            self.context = output_before_execute_tools.context
            self.lifespan.kwargs = output_before_execute_tools.kwargs
            llm_output = output_before_execute_tools.llm_output

            # ==== 第四阶段：executing_tools ====
            input_executing_tools = InputExecutingTools(
                llm_config=self.llm_config,
                context=self.context,
                llm_output=llm_output,
                kwargs=self.lifespan.kwargs
            )
            output_executing_tools: OutputExecutingTools = await self.lifespan.executing_tools.executing_tools(input_executing_tools)

            self.llm_config = output_executing_tools.llm_config
            self.context = output_executing_tools.context
            self.lifespan.kwargs = output_executing_tools.kwargs
            llm_output = output_executing_tools.llm_output
            tool_results = output_executing_tools.tool_results

            yield ToolsExecutedEvent(
                data=ToolsExecutedEvent.ToolsExecutedEventData(
                    tool_results=tool_results
                )
            )

            # ==== 第五阶段：after_execute_tools ====
            for tool_result in tool_results:
                raw_tool_result = deepcopy(tool_result)
                self.context.raw_messages.add_message(raw_tool_result)
                self.context.subsequent_messages.append(raw_tool_result)
            
            input_after_execute_tools = InputAfterExecuteTools(
                llm_config=self.llm_config,
                context=self.context, 
                llm_output=llm_output,
                tool_results=tool_results,
                kwargs=self.lifespan.kwargs
            )
            output_after_execute_tools: OutputAfterExecuteTools = await self.lifespan.after_execute_tools.after_execute_tools(input_after_execute_tools)

            self.llm_config = output_after_execute_tools.llm_config
            self.context = output_after_execute_tools.context
            self.lifespan.kwargs = output_after_execute_tools.kwargs
            llm_output = output_after_execute_tools.llm_output
            tool_results = output_after_execute_tools.tool_results

            self.context.work_messages.add_messages(tool_results)
            
        






    # ===== Lifespan API =====
    def _normalize_lifespan(self, lifespan: Optional[Lifespan]) -> Lifespan:
        """
        标准化生命周期注册类，确保每个生命周期方法都有默认实现，避免在调用时进行 None 判断

        参数列表：
        - lifespan: 用户传入的生命周期注册类实例（可选）

        返回值：
        - Lifespan: 标准化后的生命周期注册类实例，包含所有生命周期方法的默认实现
        """
        if lifespan is None:
            lifespan = Lifespan()
        if lifespan.after_user_input is None:
            lifespan.after_user_input = AfterUserInput()
        if lifespan.after_llm_output is None:
            lifespan.after_llm_output = AfterLLMOutput()
        if lifespan.before_execute_tools is None:
            lifespan.before_execute_tools = BeforeExecuteTools()
        if lifespan.executing_tools is None:
            lifespan.executing_tools = ExecutingTools()
        if lifespan.after_execute_tools is None:
            lifespan.after_execute_tools = AfterExecuteTools()
        if lifespan.after_finish is None:
            lifespan.after_finish = AfterFinish()
        return lifespan
    
    def register_lifespan(self, lifespan: Lifespan):
        """
        注册生命周期方法

        参数列表：
        - lifespan: 用户传入的生命周期注册类实例，包含用户自定义实现的生命周期方法
        """
        self.lifespan = self._normalize_lifespan(lifespan=lifespan)

    def unregister_lifespan(self):
        """
        注销生命周期方法，重置为默认实现
        """
        self.lifespan = self._normalize_lifespan(lifespan=None)

    def get_lifespan(self) -> Lifespan:
        """
        获取当前生命周期注册类实例

        返回值：
        - Lifespan: 当前生命周期注册类实例
        """
        return self.lifespan
    
    def update_lifespan(self, lifespan: Lifespan):
        """
        更新生命周期方法，允许在 Agent 运行过程中动态修改生命周期方法的实现

        参数列表：
        - lifespan: 用户传入的生命周期注册类实例，包含用户自定义实现的生命周期方法
        """
        self.register_lifespan(lifespan=lifespan)

    def update_lifespan_kwargs(self, kwargs: dict):
        """
        更新生命周期方法的扩展参数，允许在 Agent 运行过程中动态修改生命周期方法的扩展参数

        参数列表：
        - kwargs: 包含生命周期方法扩展参数的字典，这些参数会传递给生命周期方法的 data.kwargs 字段
        """
        if self.lifespan is None:
            self.lifespan = self._normalize_lifespan(lifespan=None)
        self.lifespan.kwargs.update(kwargs)

    def set_lifespan_kwargs(self):
        """
        设置生命周期方法的扩展参数
        """
        if self.lifespan is None:
            self.lifespan = self._normalize_lifespan(lifespan=None)
        self.lifespan.kwargs = {}


    # ===== 工具函数 =====
    def _wrap_to_event(self, output) -> BaseEvent:
        """将adapter的输出包装成agent输出的事件对象"""
        if isinstance(output, AssistantMessageChunk):
            # 构造 AssistantMessageChunkOutputEvent
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
            event_data = AssistantMessageChunkOutputEvent.AssistantMessageChunkOutputEventData(**data)
            return AssistantMessageChunkOutputEvent(
                data=event_data
            )

        elif isinstance(output, ToolCall):
            # 构造 ToolCallEvent
            event_data = ToolCallEvent.ToolCallEventData(
                tool_call_id=getattr(output, "tool_call_id", None),
                function_name=getattr(output, "function_name", None),
                function_args=getattr(output, "function_args", {})
            )
            return ToolCallEvent(
                data=event_data
            )

        elif isinstance(output, AssistantMessage):
            # 构造 AssistantMessageOutputEvent
            event_data = AssistantMessageOutputEvent.AssistantMessageOutputEventData(
                reasoning_content=getattr(output, "reasoning_content", None),
                content=getattr(output, "content", None),
                refusal=getattr(output, "refusal", None),
                tool_calls=getattr(output, "tool_calls", None),
                finish_reason=getattr(output, "finish_reason", "unknown"),
                token_usage=getattr(output, "token_usage", None),
                model=getattr(output, "model", None)
            )
            return AssistantMessageOutputEvent(
                data=event_data
            )

        else:
            raise ValueError(f"Unsupported output type: {type(output)}")