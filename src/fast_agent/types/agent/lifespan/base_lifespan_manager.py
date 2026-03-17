from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional, Union

from .base_lifespan import (
	IAfterExecuteTools,
	IAfterFinish,
	IAfterLLMOutput,
	IAfterUserInput,
	IBeforeExecuteTools,
	IExecutingTools,
)
from .enum.lifespan_type_enum import LifespanType


LifespanTypeLiteral = Literal[
	"after_finish",
	"after_user_input",
	"after_llm_output",
	"before_execute_tools",
	"executing_tools",
	"after_execute_tools",
]


class BaseLifespanManager(ABC):
	"""生命周期注册器抽象基类。"""

	@abstractmethod
	def __init__(
		self,
		after_finish: Optional[IAfterFinish] = None,
		after_user_input: Optional[IAfterUserInput] = None,
		after_llm_output: Optional[IAfterLLMOutput] = None,
		before_execute_tools: Optional[IBeforeExecuteTools] = None,
		executing_tools: Optional[IExecutingTools] = None,
		after_execute_tools: Optional[IAfterExecuteTools] = None,
		kwargs: Optional[Dict[str, Any]] = None,
	):
		...

	@abstractmethod
	def get_kwargs(self) -> Dict[str, Any]:
		...

	@abstractmethod
	def set_kwargs(self, kwargs: Dict[str, Any]) -> None:
		...

	@abstractmethod
	def update_kwargs(self, kwargs: Dict[str, Any]) -> None:
		...

	@abstractmethod
	def set_lifespan(
		self,
		lifespan_type: Union[LifespanType, LifespanTypeLiteral],
		handler: Any,
	) -> None:
		...

	@abstractmethod
	def get_lifespan(
		self,
		lifespan_type: Union[LifespanType, LifespanTypeLiteral],
	) -> Optional[Any]:
		...

	@abstractmethod
	def remove_lifespan(
		self,
		lifespan_type: Union[LifespanType, LifespanTypeLiteral],
	) -> None:
		...
