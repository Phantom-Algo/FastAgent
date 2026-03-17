import inspect
import uuid
from functools import wraps
from typing import Any, Callable, Optional, Union, get_type_hints

from pydantic import ConfigDict, create_model

from ...types.tool.base_tool import BaseTool
from ...types.tool.base_tool_runtime import BaseToolRuntime
from ...types.tool.domain.ask_human_policy import AskHumanPolicy
from ...types.tool.domain.guard_policy import GuardPolicy

from .tool import Tool


def tool_creator(
	func: Optional[Callable] = None,
	*,
	additional_properties: bool = False,
	strict_mode: bool = True,
	tool_name: Optional[str] = None,
	tool_description: Optional[str] = None,
	labels: Optional[list[str]] = None,
	inject_params: Optional[list[str]] = None,
	ask_human_policy: Optional[AskHumanPolicy] = None,
	guard_policy: Optional[GuardPolicy] = None,
) -> Callable:
	"""
	装饰器 tool_creator：将函数包装为可供 AI 调用的 Tool 对象。

	支持两种调用方式：
	1. @tool_creator
	2. @tool_creator(tool_name="xxx", strict_mode=False)
	"""

	def decorator(f: Callable) -> BaseTool:
		original_func = inspect.unwrap(f)
		signature = inspect.signature(original_func)
		type_hints = get_type_hints(original_func, include_extras=True)

		normalized_inject_params = list(inject_params or [])

		# 动态构建 Pydantic 参数模型
		fields_defs: dict[str, tuple[Any, Any]] = {}
		tool_runtime_param_name = None
		for param_name, param in signature.parameters.items():
			if param_name in ("self", "cls"):
				continue

			if param_name in normalized_inject_params:
				continue
			
			if type_hints.get(param_name) == BaseToolRuntime:
				if tool_runtime_param_name is not None:
					raise ValueError("Error: Multiple parameters annotated with BaseToolRuntime are not allowed.")
				tool_runtime_param_name = param_name
				continue
			   
			annotation = type_hints.get(param_name, Any)
			default_value = param.default

			# strict_mode 开启时，所有参数都按 required 处理
			if strict_mode or default_value is inspect.Parameter.empty:
				default_value = ...

			fields_defs[param_name] = (annotation, default_value)

		model_name = "".join(word.capitalize() for word in original_func.__name__.split("_")) + "ToolParamsModel"
		model_config = ConfigDict(extra="allow" if additional_properties else "forbid")
		tool_params_model = create_model(model_name, __config__=model_config, **fields_defs)

		# 异步
		@wraps(f)
		async def async_wrapper(*args, **kwargs):
			return await f(*args, **kwargs)

		# 同步
		@wraps(f)
		def sync_wrapper(*args, **kwargs):
			return f(*args, **kwargs)

		is_async = inspect.iscoroutinefunction(f)
		wrapper = async_wrapper if is_async else sync_wrapper

		final_name = tool_name if tool_name else original_func.__name__

		final_ask_human_policy = ask_human_policy or AskHumanPolicy(timeout=300)
		final_guard_policy = guard_policy

		return Tool(
			name=final_name,
			description=tool_description if tool_description else inspect.getdoc(original_func) or "",
			args_schema=tool_params_model,
			func=wrapper,
			is_async=is_async,
			labels=list(labels or []),
			inject_params=normalized_inject_params,
			tool_runtime_param_name=tool_runtime_param_name,
			ask_human_policy=final_ask_human_policy,
			guard_policy=final_guard_policy,
		)

	if func is None:
		return decorator

	return decorator(func)


def _clean_tool_schema(schema: Union[dict, list]) -> Union[dict, list]:
	"""递归清洗 schema 中不必要的 title 字段。"""
	if isinstance(schema, dict):
		schema.pop("title", None)
		for value in schema.values():
			_clean_tool_schema(value)
	elif isinstance(schema, list):
		for item in schema:
			_clean_tool_schema(item)
	return schema
