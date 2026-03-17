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

# 动态模型注册表，使 pickle 能够找到 create_model 生成的类
_DYNAMIC_MODEL_REGISTRY = {}


def __getattr__(name):
	"""模块级 __getattr__，让 pickle 能通过模块属性查找动态创建的模型类。"""
	if name in _DYNAMIC_MODEL_REGISTRY:
		return _DYNAMIC_MODEL_REGISTRY[name]
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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

		# 注册动态模型，使其可被 pickle 序列化
		tool_params_model.__module__ = __name__
		tool_params_model.__qualname__ = model_name
		_DYNAMIC_MODEL_REGISTRY[model_name] = tool_params_model

		# 生成唯一的 wrapper 名称，避免与装饰后的 Tool 对象冲突
		wrapper_name = f"_wrapper_{original_func.__name__}_{uuid.uuid4().hex[:8]}"

		# 异步
		async def async_wrapper(*args, **kwargs):
			return await f(*args, **kwargs)

		# 同步
		def sync_wrapper(*args, **kwargs):
			return f(*args, **kwargs)

		is_async = inspect.iscoroutinefunction(f)
		wrapper = async_wrapper if is_async else sync_wrapper

		# 注册 wrapper 函数，使其可被 pickle 序列化
		wrapper.__name__ = wrapper_name
		wrapper.__qualname__ = wrapper_name
		wrapper.__module__ = __name__
		_DYNAMIC_MODEL_REGISTRY[wrapper_name] = wrapper

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
