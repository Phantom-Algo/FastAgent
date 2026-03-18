from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..tool.base_tool import BaseTool


class BaseMCPAdapter(ABC):
	"""MCP 适配器抽象层：负责把 MCP 工具转换为框架内的 BaseTool。"""


	@abstractmethod
	async def fetch_tools(self, server_name: str, server_config: Dict[str, Any]) -> List[BaseTool]:
		"""根据 server 配置拉取工具并完成 BaseTool 适配。"""
		...

	@abstractmethod
	async def fetch_tools_via_stdio(self, server_name: str, server_config: Dict[str, Any]) -> List[BaseTool]:
		"""通过 stdio 方式拉取工具。"""
		...

	@abstractmethod
	async def fetch_tools_via_sse(self, server_name: str, server_config: Dict[str, Any]) -> List[BaseTool]:
		"""通过 sse 方式拉取工具。"""
		...

	@abstractmethod
	async def fetch_tools_via_streamablehttp(self, server_name: str, server_config: Dict[str, Any]) -> List[BaseTool]:
		"""通过 streamablehttp 方式拉取工具。"""
		...