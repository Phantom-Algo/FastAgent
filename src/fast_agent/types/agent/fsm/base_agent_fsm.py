from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, TYPE_CHECKING

from .dto.agent_fsm_shared_data import AgentFSMSharedData
from ..event.base_event import BaseEvent

if TYPE_CHECKING:
    from .base_agent_fsm_state import BaseAgentFSMState

class BaseAgentFSM(ABC):
	"""BaseAgentFSM 定义 Agent 状态机驱动器的抽象基类。"""

	def __init__(
		self,
		initial_state: "BaseAgentFSMState",
		shared_data: AgentFSMSharedData
	):
		# 状态变换共享数据（上下文）
		self.shared_data = shared_data
		
        # 状态机当前状态和下一个状态
		self.current_state: Optional["BaseAgentFSMState"] = initial_state
		self.next_state: Optional["BaseAgentFSMState"] = None


	@abstractmethod
	async def run(self) -> AsyncGenerator[BaseEvent, None]:
		"""运行状态机主循环并持续产出事件流。"""
		...
