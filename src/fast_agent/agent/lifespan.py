from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List
from fast_agent.llm import LLMConfig, Context, UserMessage, AssistantMessage, ToolResultMessage

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

# ===== lifespan 注册类 =====
class Lifespan:
    """
    Lifespan 生命周期注册类
    """
    def __init__(
        self,
        after_finish: Optional[IAfterFinish] = None,
        after_user_input: Optional[IAfterUserInput] = None,
        after_llm_output: Optional[IAfterLLMOutput] = None,
        before_execute_tools: Optional[IBeforeExecuteTools] = None,
        executing_tools: Optional[IExecutingTools] = None,
        after_execute_tools: Optional[IAfterExecuteTools] = None,
        kwargs: Optional[Dict[str, Any]] = None
    ):
        self.after_finish = after_finish
        self.after_user_input = after_user_input
        self.after_llm_output = after_llm_output
        self.before_execute_tools = before_execute_tools
        self.executing_tools = executing_tools
        self.after_execute_tools = after_execute_tools
        self.kwargs = kwargs or {}
