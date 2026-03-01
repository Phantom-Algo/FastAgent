from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, Callable, Optional, get_type_hints

from pydantic import create_model

from .schema.BaseTool import BaseTool


def tool(
    func: Optional[Callable] = None,
    *,
    tool_name: Optional[str] = None,
    description: Optional[str] = None,
    labels: Optional[list[str]] = None,
    inject_params: Optional[list[str]] = None,
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
        for param_name, param in signature.parameters.items():
            if param_name in ("self", "cls"):
                continue
            if param_name in inject_set:
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
        )

    return decorator if func is None else decorator(func)