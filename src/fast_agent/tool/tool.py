from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, Callable, Optional, get_type_hints, TYPE_CHECKING, Dict

from pydantic import create_model, BaseModel, Field, ConfigDict

from .schema.base_tool import BaseTool
from .human_review import HumanReviewChannel

if TYPE_CHECKING:
    from fast_agent.llm import LLMConfig, Context, AssistantMessage, UserMessage, ToolResultMessage
    from .guard import GuardInfo, GuardRequestSchema
    LLMConfigType = LLMConfig
    ContextType = Context
    AssistantMessageType = AssistantMessage
    UserMessageType = UserMessage
else:
    LLMConfigType = Any
    ContextType = Any
    AssistantMessageType = Any
    UserMessageType = Any

class ToolRuntime(BaseModel):
    """
    ToolRuntime 包含工具执行时的上下文信息，供工具函数使用。
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool_call_id: str
    this_tool: Optional[BaseTool] = None
    llm_config: Optional[LLMConfigType] = None
    context: Optional[ContextType] = None
    llm_output: Optional[AssistantMessageType] = None
    user_input: Optional[UserMessageType] = None
    human_review_channel: Optional[HumanReviewChannel] = None
    human_review_timeout: Optional[int] = None
    kwargs: Dict[str, Any] = Field(default_factory=dict)

    async def human_review(self, data: dict, timeout: Optional[int] = None) -> dict:
        """
        发起人工审核请求，并等待响应。
        若正常响应，则会返回用户响应的字典
        若发生错误，以下为建议开发者考虑的处理方式：
        - 超时错误：建议开发者在调用时捕获超时异常，在设置合理的重试后若依旧超时，则封装合理的ToolResultMessage返回错误信息
        TODO：添加更加详细的使用说明
        """
        return await self.human_review_channel.request(data, timeout=timeout or self.human_review_timeout)



def tool(
    func: Optional[Callable] = None,
    *,
    tool_name: Optional[str] = None,
    description: Optional[str] = None,
    labels: Optional[list[str]] = None,
    inject_params: Optional[list[str]] = None,
    human_review_timeout: Optional[int] = None,
    guard_info: Optional[GuardInfo] = None,
    guard: Optional[Callable[..., bool]] = None,
    reject_response: Optional[Callable[[GuardRequestSchema], "ToolResultMessage"]] = None
) -> Callable:
    """
    将普通函数/异步函数包装为 BaseTool。

    支持两种使用方式：
    1) `@tool`
    2) `@tool(tool_name="xxx", inject_params=["ctx"])`

    参数说明：
    - tool_name: 工具名，默认使用函数名
    - description: 工具描述，默认读取函数 docstring
    - labels: 工具标签
    - inject_params: 由运行环境注入的参数名列表，这些参数不会出现在 args_schema 中
    - human_review_timeout: 人工审核超时时间（秒）
    - guard: 工具调用护栏函数，用于在调用工具前进行条件检查
    """

    def decorator(target: Callable) -> BaseTool:
        original_func = inspect.unwrap(target)
        signature = inspect.signature(original_func)
        type_hints = get_type_hints(original_func, include_extras=True)

        inject_set = set(inject_params or [])
        unknown_inject_params = sorted(inject_set - set(signature.parameters.keys()))
        if unknown_inject_params:
            raise ValueError(
                f"inject_params 包含不存在于函数签名中的参数: {unknown_inject_params}"
            )

        fields_defs: dict[str, tuple[Any, Any]] = {}
        tool_runtime_name = ""
        for param_name, param in signature.parameters.items():
            # --- 过滤掉特殊的参数 ---
            if param_name in ("self", "cls"):
                continue
            if param_name in inject_set:
                continue
            if type_hints.get(param_name, Any) == ToolRuntime:
                if tool_runtime_name:
                    raise ValueError("Not allow more than one ToolRuntime parameters in tool function.")
                tool_runtime_name = param_name
                continue

            annotation = type_hints.get(param_name, Any)
            default_value = param.default
            if default_value is inspect.Parameter.empty:
                default_value = ...

            fields_defs[param_name] = (annotation, default_value)

        model_name = "".join(part.capitalize() for part in original_func.__name__.split("_")) + "Args"
        args_schema = create_model(model_name, **fields_defs)

        tool_desc = description if description is not None else (inspect.getdoc(original_func) or "")
        is_async = inspect.iscoroutinefunction(target)

        @wraps(target)
        async def async_wrapper(*args, **kwargs):
            return await target(*args, **kwargs)

        @wraps(target)
        def sync_wrapper(*args, **kwargs):
            return target(*args, **kwargs)

        wrapper = async_wrapper if is_async else sync_wrapper

        return BaseTool(
            name=tool_name or original_func.__name__,
            description=tool_desc,
            args_schema=args_schema,
            func=wrapper,
            is_async=is_async,
            labels=labels or [],
            inject_params=list(inject_set),
            tool_runtime_name=tool_runtime_name or None,
            human_review_timeout=human_review_timeout,
            guard=guard,
            guard_info=guard_info,
            reject_response=reject_response,
        )

    return decorator if func is None else decorator(func)