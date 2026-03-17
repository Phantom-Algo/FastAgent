from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator, TYPE_CHECKING

from ..event.base_event import BaseEvent
from .enum.agent_fsm_state_enum import AgentFSMStateEnum

if TYPE_CHECKING:
	from .base_agent_fsm import BaseAgentFSM


class BaseAgentFSMState(ABC):
	"""BaseAgentFSMState 定义状态机状态节点的统一抽象接口。"""

	@abstractmethod
	def get_status(self) -> AgentFSMStateEnum:
		"""返回当前状态节点的状态标识。"""
		...

	@abstractmethod
	async def execute(self, fsm: "BaseAgentFSM") -> AsyncGenerator[BaseEvent, None]:
		"""
		执行当前状态逻辑并持续产出事件。

		约定：
		- 状态执行结束后由实现方设置 fsm.next_state
		- 当 fsm.next_state 为 None 时，状态机终止
		"""
		...
