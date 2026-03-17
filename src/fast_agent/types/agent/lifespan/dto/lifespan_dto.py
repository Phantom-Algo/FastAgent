from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, ConfigDict, Field

from ....context.base_context import BaseContext
from ....llm.base_llm_config import BaseLLMConfig
from ....messages.domain.assistant_message import AssistantMessage, ToolCall
from ....messages.domain.tool_result_message import ToolResultMessage
from ....messages.domain.user_message import UserMessage
from ...event.base_event_channel import BaseEventChannel
from ....tool.domain.guard_policy import GuardPolicyHumanResponseSchema


class BaseLifespanData(BaseModel):
	"""生命周期阶段通用数据对象。"""

	model_config = ConfigDict(arbitrary_types_allowed=True)

	llm_config: BaseLLMConfig
	
	context: BaseContext
	
	event_channel: BaseEventChannel
	
	kwargs: Dict[str, Any] = Field(default_factory=dict)
	
	user_input: Optional[UserMessage] = None
	
	llm_output: Optional[AssistantMessage] = None
	

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
		


class AfterFinishRequest(BaseLifespanData):
	"""after_finish 请求数据。"""


class AfterFinishResponse(BaseLifespanData):
	"""after_finish 响应数据。"""


class AfterUserInputRequest(BaseLifespanData):
	"""after_user_input 请求数据。"""


class AfterUserInputResponse(BaseLifespanData):
	"""after_user_input 响应数据。"""


class AfterLLMOutputRequest(BaseLifespanData):
	"""after_llm_output 请求数据。"""


class AfterLLMOutputResponse(BaseLifespanData):
	"""after_llm_output 响应数据。"""


class BeforeExecuteToolsRequest(BaseLifespanData):
	"""before_execute_tools 请求数据。"""
	human_response: Optional[Dict[str, GuardPolicyHumanResponseSchema]] = None


class BeforeExecuteToolsResponse(BaseLifespanData):
	"""before_execute_tools 响应数据。"""
	pending_tool_calls: List[ToolCall] = Field(default_factory=list)
	finished_tool_calls: List[ToolCall] = Field(default_factory=list)
	prebuilt_tool_results: List[ToolResultMessage] = Field(default_factory=list)


class ExecutingToolsRequest(BaseLifespanData):
	"""executing_tools 请求数据。"""
	pending_tool_calls: List[ToolCall] = Field(default_factory=list)
	finished_tool_calls: List[ToolCall] = Field(default_factory=list)
	prebuilt_tool_results: List[ToolResultMessage] = Field(default_factory=list)


class ExecutingToolsResponse(BaseLifespanData):
	"""executing_tools 响应数据。"""

	finished_tool_calls: List[ToolCall] = Field(default_factory=list)
	tool_results: List[ToolResultMessage] = Field(default_factory=list)


class AfterExecuteToolsRequest(BaseLifespanData):
	"""after_execute_tools 请求数据。"""

	tool_results: List[ToolResultMessage] = Field(default_factory=list)


class AfterExecuteToolsResponse(BaseLifespanData):
	"""after_execute_tools 响应数据。"""

	tool_results: List[ToolResultMessage] = Field(default_factory=list)
