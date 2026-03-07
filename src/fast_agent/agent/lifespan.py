from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Union, Literal, Tuple
from fast_agent.llm import LLMConfig, Context, UserMessage, AssistantMessage, ToolResultMessage, ToolCall
from fast_agent.tool import ToolRuntime, HumanReviewChannel, BaseTool, ToolCallGuardTriggered, GuardRequestSchema, GuardTriggeredToolCallContext
from .event import EventChannel
import asyncio

# ======= 生命周期类型枚举 =======
class LifespanType(Enum):
    AFTER_FINISH = "after_finish"
    AFTER_USER_INPUT = "after_user_input"
    AFTER_LLM_OUTPUT = "after_llm_output"
    BEFORE_EXECUTE_TOOLS = "before_execute_tools"
    EXECUTING_TOOLS = "executing_tools"
    AFTER_EXECUTE_TOOLS = "after_execute_tools"

# ======= 生命周期数据传输对象 =======

@dataclass
class BaseLifespanData:
    """
    BaseLifespanData 生命周期基础数据传输对象

    - kwargs 用于在生命周期全流程中传递开发者自定义字段。
    - 若希望跨阶段共享状态，请在调度入口传入同一个 kwargs dict 引用。
    """

    llm_config: LLMConfig
    context: Context
    event_channel: EventChannel
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def get_kwarg(self, key: str, default: Any = None) -> Any:
        return self.kwargs.get(key, default)

    def set_kwarg(self, key: str, value: Any) -> None:
        self.kwargs[key] = value

    def pop_kwarg(self, key: str, default: Any = None) -> Any:
        return self.kwargs.pop(key, default)

    def require_kwarg(self, key: str) -> Any:
        if key not in self.kwargs:
            raise KeyError(f"Missing required kwargs key: {key}")
        return self.kwargs[key]

    def update_agent(self, agent: Any) -> Any:
        """
        将当前生命周期数据对象中的通用状态回写到 Agent。

        回写字段：
        - llm_config
        - context
        - lifespan.kwargs
        """
        agent.llm_config = self.llm_config
        agent.context = self.context
        agent.lifespan.kwargs = self.kwargs
        return agent

    @classmethod
    def from_agent(cls, agent: Any, **extra_fields: Any):
        """
        从 Agent 当前状态构建生命周期数据对象。

        参数列表：
        - agent: Agent 实例（需要包含 llm_config / context / lifespan.kwargs）
        - extra_fields: 生命周期数据对象的额外字段（如 user_input / llm_output / tool_results）
        """
        return cls(
            llm_config=agent.llm_config,
            context=agent.context,
            kwargs=agent.lifespan.kwargs,
            event_channel=agent.event_channel,
            **extra_fields,
        )

    @staticmethod
    def create(data_cls: type["BaseLifespanData"], agent: Any, **extra_fields: Any):
        """
        静态工厂：按指定数据类从 Agent 当前状态创建生命周期数据对象。
        """
        return data_cls.from_agent(agent, **extra_fields)


@dataclass
class InputAfterFinish(BaseLifespanData):
    """
    InputAfterFinish 生命周期数据传输对象
    包含 after_finish 生命周期方法所需的参数
    """


@dataclass
class OutputAfterFinish(BaseLifespanData):
    """
    OutputAfterFinish 生命周期数据传输对象
    包含 after_finish 生命周期方法的返回值参数
    """


@dataclass
class InputAfterUserInput(BaseLifespanData):
    """
    InputAfterUserInput 生命周期数据传输对象
    包含 after_user_input 生命周期方法所需的参数
    """

    user_input: Optional[UserMessage] = None


@dataclass
class OutputAfterUserInput(BaseLifespanData):
    """
    OutputAfterUserInput 生命周期数据传输对象
    包含 after_user_input 生命周期方法的返回值参数
    """

    user_input: Optional[UserMessage] = None


@dataclass
class InputAfterLLMOutput(BaseLifespanData):
    """
    InputAfterLLMOutput 生命周期数据传输对象
    包含 after_llm_output 生命周期方法所需的参数
    """
    llm_output: Optional[AssistantMessage] = None


@dataclass
class OutputAfterLLMOutput(BaseLifespanData):
    """
    OutputAfterLLMOutput 生命周期数据传输对象
    包含 after_llm_output 生命周期方法的返回值参数
    """
    llm_output: Optional[AssistantMessage] = None


@dataclass
class InputBeforeExecuteTools(BaseLifespanData):
    """
    InputBeforeExecuteTools 生命周期数据传输对象
    包含 before_execute_tools 生命周期方法所需的参数
    """
    llm_output: Optional[AssistantMessage] = None


@dataclass
class OutputBeforeExecuteTools(BaseLifespanData):
    """
    OutputBeforeExecuteTools 生命周期数据传输对象
    包含 before_execute_tools 生命周期方法的返回值参数
    """
    llm_output: Optional[AssistantMessage] = None


@dataclass
class InputExecutingTools(BaseLifespanData):
    """
    InputExecutingTools 生命周期数据传输对象
    包含 executing_tools 生命周期方法所需的参数（工具执行阶段，可多次触发）
    """
    llm_output: Optional[AssistantMessage] = None
    finished_tool_calls: List[ToolCall] = field(default_factory=list)  # 已完成工具调用列表
    human_response: Optional[Dict[str, GuardRequestSchema]] = None  # 人类响应数据, key 为 tool_call_id，value为响应内容


@dataclass
class OutputExecutingTools(BaseLifespanData):
    """
    OutputExecutingTools 生命周期数据传输对象
    包含 executing_tools 生命周期方法的返回值参数
    """

    llm_output: Optional[AssistantMessage] = None
    tool_results: List[ToolResultMessage] = field(default_factory=list)


@dataclass
class InputAfterExecuteTools(BaseLifespanData):
    """
    InputAfterExecuteTools 生命周期数据传输对象
    包含 after_execute_tools 生命周期方法所需的参数
    """

    llm_output: Optional[AssistantMessage] = None
    tool_results: List[ToolResultMessage] = field(default_factory=list)


@dataclass
class OutputAfterExecuteTools(BaseLifespanData):
    """
    OutputAfterExecuteTools 生命周期数据传输对象
    包含 after_execute_tools 生命周期方法的返回值参数
    """

    llm_output: Optional[AssistantMessage] = None
    tool_results: List[ToolResultMessage] = field(default_factory=list)


# ======= 生命周期接口 =======

class IAfterFinish(ABC):
    """
    IAfterFinish 生命周期接口
    定义在用户输入前 / 完整一轮交互结束后的生命周期方法
    """

    @abstractmethod
    async def after_finish(self, data: InputAfterFinish) -> OutputAfterFinish:
        """
        after_finish 生命周期方法

        参数列表：
        - data [InputAfterFinish]: 包含 after_finish 生命周期方法所需的参数
            - llm_config: LLMConfig 大模型配置
            - context: Context 上下文信息
            - kwargs: 开发者自定义扩展字段（可跨阶段共享）
        """
        pass


class IAfterUserInput(ABC):
    """
    IAfterUserInput 生命周期接口
    定义在用户输入后 / 大模型生成结果前的生命周期方法
    """

    @abstractmethod
    async def after_user_input(self, data: InputAfterUserInput) -> OutputAfterUserInput:
        """
        after_user_input 生命周期方法

        参数列表：
        - data [InputAfterUserInput]: 包含 after_user_input 生命周期方法所需的参数
            - llm_config: LLMConfig 大模型配置
            - context: Context 上下文信息
            - user_input: UserMessage 用户输入消息对象
            - kwargs: 开发者自定义扩展字段（可跨阶段共享）
        """
        pass


class IAfterLLMOutput(ABC):
    """
    IAfterLLMOutput 生命周期接口
    定义在大模型生成结果后，路由到“结束”或“工具调用”分支前的生命周期方法
    """

    @abstractmethod
    async def after_llm_output(self, data: InputAfterLLMOutput) -> OutputAfterLLMOutput:
        """
        after_llm_output 生命周期方法

        参数列表：
        - data [InputAfterLLMOutput]: 包含 after_llm_output 生命周期方法所需的参数
            - llm_config: LLMConfig 大模型配置
            - context: Context 上下文信息
            - llm_output: AssistantMessage 大模型输出消息对象
            - kwargs: 开发者自定义扩展字段（可跨阶段共享）
        """
        pass


class IBeforeExecuteTools(ABC):
    """
    IBeforeExecuteTools 生命周期接口
    定义在路由到“工具调用”分支之后、执行工具调用之前的生命周期方法
    """

    @abstractmethod
    async def before_execute_tools(self, data: InputBeforeExecuteTools) -> OutputBeforeExecuteTools:
        """
        before_execute_tools 生命周期方法

        参数列表：
        - data [InputBeforeExecuteTools]: 包含 before_execute_tools 生命周期方法所需的参数
            - llm_config: LLMConfig 大模型配置
            - context: Context 上下文信息
            - llm_output: AssistantMessage 大模型输出消息对象
            - kwargs: 开发者自定义扩展字段（可跨阶段共享）
        """
        pass


class IExecutingTools(ABC):
    """
    IExecutingTools 生命周期接口
    定义在工具执行阶段的生命周期方法（可在每个工具执行过程中多次触发）
    """

    @abstractmethod
    async def executing_tools(self, data: InputExecutingTools) -> OutputExecutingTools:
        """
        executing_tools 生命周期方法

        参数列表：
        - data [InputExecutingTools]: 包含 executing_tools 生命周期方法所需的参数
            - llm_config: LLMConfig 大模型配置
            - context: Context 上下文信息
            - llm_output: AssistantMessage 大模型输出消息对象
            - kwargs: 开发者自定义扩展字段（可跨阶段共享）
        """
        pass


class IAfterExecuteTools(ABC):
    """
    IAfterExecuteTools 生命周期接口
    定义在执行工具调用之后、将工具调用结果继续路由到后续LLM处理之前的生命周期方法
    """

    @abstractmethod
    async def after_execute_tools(self, data: InputAfterExecuteTools) -> OutputAfterExecuteTools:
        """
        after_execute_tools 生命周期方法

        参数列表：
        - data [InputAfterExecuteTools]: 包含 after_execute_tools 生命周期方法所需的参数
            - llm_config: LLMConfig 大模型配置
            - context: Context 上下文信息
            - llm_output: AssistantMessage 大模型输出消息对象
            - tool_results: List[ToolResultMessage] 工具调用结果消息对象列表
            - kwargs: 开发者自定义扩展字段（可跨阶段共享）
        """
        pass


# ======= 生命周期默认实现 =======
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

            # --- 注入开发者指定的运行时注入参数 ---
            for param_name in (tool.inject_params or []):
                if param_name in tool_inject_params:
                    call_kwargs[param_name] = tool_inject_params[param_name]

            # --- 系统自动注入 tool_runtime 参数（包含工具调用的上下文信息，供工具函数使用） ---
            tool_runtime_name = tool.get_tool_runtime_name()
            if tool_runtime_name:
                call_kwargs[tool_runtime_name] = ToolRuntime(
                    tool_call_id=tool_call.tool_call_id,
                    this_tool=tool,
                    llm_config=data.llm_config,
                    context=data.context,
                    llm_output=llm_output,
                    human_review_channel=HumanReviewChannel(
                        event_channel=data.event_channel,
                    ),
                    human_review_timeout=tool.human_review_timeout,
                    kwargs=data.kwargs,
                )

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
            
        # 过滤
        passed_calls, filtered_calls, rejected_calls = self._filter_tool_calls_by_guard(tool_calls, tools_by_name, data.human_response)

        # gather 返回结果顺序与传入顺序一致，可同时执行异步/同步工具调用
        # TODO:处理一些工具 raise 异常的边缘情况 
        tool_results = await asyncio.gather(*[_run_single_tool_call(tool_call) for tool_call in passed_calls]) if passed_calls else []

        # rejected_calls 需要封装成 ToolResultMessage 返回给 LLM，供后续处理
        for rejected_call in rejected_calls:
            tool = tools_by_name.get(rejected_call.function_name)
            reject_response_func = tool.reject_response if tool and tool.reject_response else lambda a: ToolResultMessage(
                tool_call_id=rejected_call.tool_call_id,
                name=rejected_call.function_name,
                content=f"工具 `{rejected_call.function_name}` 调用被拒绝。",
                is_error=False
            )
            tool_results.append(reject_response_func(data.human_response.get(rejected_call.tool_call_id)))

        # filtered_calls 需发起人工审核请求
        if filtered_calls:
            raise ToolCallGuardTriggered(
                "Toolcall guard has been triggered",
                contexts=[
                    GuardTriggeredToolCallContext(
                        tool_call=tool_call,
                        tool=tools_by_name.get(tool_call.function_name),
                    )
                    for tool_call in filtered_calls
                ],
                finished_tool_calls=data.finished_tool_calls.extend(passed_calls)
            )

        return OutputExecutingTools(
            llm_config=data.llm_config,
            context=data.context,
            llm_output=llm_output,
            tool_results=tool_results,
            kwargs=data.kwargs
        )
    
    def _filter_tool_calls_by_guard(
        self, 
        tool_calls: List[ToolCall], 
        tools_by_name: Dict[str, BaseTool], 
        finished_tool_calls: List[ToolCall],
        human_response: Optional[Dict[str, GuardRequestSchema]] = None
    ) -> Tuple[List[ToolCall], List[ToolCall], List[ToolCall]]:
        """
        基于guard的工具调用过滤器
        - 返回两个列表：第一个是通过过滤即将执行的工具调用列表，第二个是被过滤掉需要等待人工审查的工具调用列表，第三个是被拒绝的工具调用列表（不再被执行，而是需要后续包装成 ToolResultMessage 返回给 LLM）
        """
        passed_calls = []
        filtered_calls = []
        rejected_calls = []

        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.function_name)

            # 检测到该工具有护栏
            if tool and tool.guard:
                if not human_response or (human_response and tool_call.tool_call_id not in human_response):
                    # 如果没有响应数据，或者有响应数据但当前工具调用没有对应的响应，则加入被过滤的工具调用列表中，用于后续发起人工审核请求
                    filtered_calls.append(tool_call)
                else:
                    # 有响应数据且当前工具调用有对应响应，使用护栏函数判断是否通过
                    response_data = human_response[tool_call.tool_call_id]
                    if tool.guard(response_data):
                        passed_calls.append(tool_call)
                    else:
                        # 护栏函数判断不通过
                        rejected_calls.append(tool_call)

            elif tool not in finished_tool_calls:
                # 没有护栏且尚未被执行，直接加入通过的工具调用列表中
                passed_calls.append(tool_call)

        return passed_calls, filtered_calls, rejected_calls




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

# ===== lifespan 注册类 =====
class Lifespan:
    """
    Lifespan 生命周期注册类
    """
    def __init__(
        self,
        after_finish: IAfterFinish = AfterFinish(),
        after_user_input: IAfterUserInput = AfterUserInput(),
        after_llm_output: IAfterLLMOutput = AfterLLMOutput(),
        before_execute_tools: IBeforeExecuteTools = BeforeExecuteTools(),
        executing_tools: IExecutingTools = ExecutingTools(),
        after_execute_tools: IAfterExecuteTools = AfterExecuteTools(),
        kwargs: Optional[Dict[str, Any]] = None
    ):
        self.after_finish = after_finish
        self.after_user_input = after_user_input
        self.after_llm_output = after_llm_output
        self.before_execute_tools = before_execute_tools
        self.executing_tools = executing_tools
        self.after_execute_tools = after_execute_tools
        self.kwargs = kwargs or {}

    def get_kwargs(self) -> Dict[str, Any]:
        return self.kwargs
    
    def set_kwargs(self, kwargs: Dict[str, Any]) -> None:
        self.kwargs = kwargs

    def update_kwargs(self, kwargs: Dict[str, Any]) -> None:
        self.kwargs.update(kwargs)

    

    def set_lifespan(self, type: Union[LifespanType, Literal["after_finish", "after_user_input", "after_llm_output", "before_execute_tools", "executing_tools", "after_execute_tools"]], handler: Any) -> None:
        if isinstance(type, str):
            type = LifespanType(type)
        
        if type == LifespanType.AFTER_FINISH:
            self.after_finish = handler
        elif type == LifespanType.AFTER_USER_INPUT:
            self.after_user_input = handler
        elif type == LifespanType.AFTER_LLM_OUTPUT:
            self.after_llm_output = handler
        elif type == LifespanType.BEFORE_EXECUTE_TOOLS:
            self.before_execute_tools = handler
        elif type == LifespanType.EXECUTING_TOOLS:
            self.executing_tools = handler
        elif type == LifespanType.AFTER_EXECUTE_TOOLS:
            self.after_execute_tools = handler

    def get_lifespan(self, type: Union[LifespanType, Literal["after_finish", "after_user_input", "after_llm_output", "before_execute_tools", "executing_tools", "after_execute_tools"]]) -> Optional[Any]:
        if isinstance(type, str):
            type = LifespanType(type)
        
        if type == LifespanType.AFTER_FINISH:
            return self.after_finish
        elif type == LifespanType.AFTER_USER_INPUT:
            return self.after_user_input
        elif type == LifespanType.AFTER_LLM_OUTPUT:
            return self.after_llm_output
        elif type == LifespanType.BEFORE_EXECUTE_TOOLS:
            return self.before_execute_tools
        elif type == LifespanType.EXECUTING_TOOLS:
            return self.executing_tools
        elif type == LifespanType.AFTER_EXECUTE_TOOLS:
            return self.after_execute_tools
        
        return None
    
    def remove_lifespan(self, type: Union[LifespanType, Literal["after_finish", "after_user_input", "after_llm_output", "before_execute_tools", "executing_tools", "after_execute_tools"]]) -> None:
        if isinstance(type, str):
            type = LifespanType(type)
        
        if type == LifespanType.AFTER_FINISH:
            self.after_finish = AfterFinish()
        elif type == LifespanType.AFTER_USER_INPUT:
            self.after_user_input = AfterUserInput()
        elif type == LifespanType.AFTER_LLM_OUTPUT:
            self.after_llm_output = AfterLLMOutput()
        elif type == LifespanType.BEFORE_EXECUTE_TOOLS:
            self.before_execute_tools = BeforeExecuteTools()
        elif type == LifespanType.EXECUTING_TOOLS:
            self.executing_tools = ExecutingTools()
        elif type == LifespanType.AFTER_EXECUTE_TOOLS:
            self.after_execute_tools = AfterExecuteTools()
