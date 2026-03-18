from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..tool.base_tool import BaseTool
from ..tool.domain.ask_human_policy import AskHumanPolicy
from ..tool.domain.guard_policy import GuardPolicy


class BaseMCPManager(ABC):
	"""
	MCP 工具管理器抽象层。

	负责：
	1. 从 MCP Server 拉取并注册工具
	2. 对注册后的 BaseTool 做增强（description / labels / guard 等）
	3. 提供基础 CRUD
	"""

	@abstractmethod
	async def register_servers_from_json(
		self,
		config_json: str,
		tool_enhancements: Optional[Dict[str, Dict[str, Any]]] = None,
		clear_existing: bool = False,
	) -> List[BaseTool]:
		...

	@abstractmethod
	async def register_servers_from_addresses(
		self,
		addresses: List[str],
		tool_enhancements: Optional[Dict[str, Dict[str, Any]]] = None,
		clear_existing: bool = False,
	) -> List[BaseTool]:
		...

	@abstractmethod
	def enhance_tool_by_id(
		self,
		id: str,
		description: Optional[str] = None,
		labels: Optional[List[str]] = None,
		guard_policy: Optional[GuardPolicy] = None,
		ask_human_policy: Optional[AskHumanPolicy] = None,
	) -> bool:
		...

	# === 增 ===
	@abstractmethod
	def add_tool(self, tool: BaseTool) -> None:
		...

	# === 删 ===
	@abstractmethod
	def remove_tool_by_id(self, id: str) -> Optional[BaseTool]:
		...

	@abstractmethod
	def clear_tools(self) -> None:
		...

	# === 改 ===
	@abstractmethod
	def update_tool_by_id(self, id: str, new_tool: BaseTool) -> bool:
		...

	@abstractmethod
	def update_tools(self, new_tools: List[BaseTool]) -> None:
		...

	# === 查 ===
	@abstractmethod
	def get_tools(self) -> List[BaseTool]:
		...

	@abstractmethod
	def get_tool_count(self) -> int:
		...

	@abstractmethod
	def get_tool_by_id(self, id: str) -> Optional[BaseTool]:
		...

	@abstractmethod
	def get_server_configs(self) -> Dict[str, Dict[str, Any]]:
		...

