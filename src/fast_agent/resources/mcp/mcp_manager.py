from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen

from ...types.mcp.base_mcp_adapter import BaseMCPAdapter
from ...types.mcp.base_mcp_manager import BaseMCPManager
from ...types.tool.base_tool import BaseTool
from ...types.tool.domain.ask_human_policy import AskHumanPolicy
from ...types.tool.domain.guard_policy import GuardPolicy
from .mcp_adapter import MCPAdapter


class MCPManager(BaseMCPManager):
	"""MCP 工具管理器：负责工具注册、增强和基础 CRUD。"""

	def __init__(
		self,
		adapter: Optional[BaseMCPAdapter] = None,
		tools: Optional[List[BaseTool]] = None,
	):
		self.adapter = adapter if adapter is not None else MCPAdapter()
		self.tools = list(tools or [])
		self.server_configs: Dict[str, Dict[str, Any]] = {}

	async def register_servers_from_json(
		self,
		config_json: str,
		tool_enhancements: Optional[Dict[str, Dict[str, Any]]] = None,
		clear_existing: bool = False,
	) -> List[BaseTool]:
		config_data = self._parse_json_text(config_json)
		server_map = self._extract_server_map(config_data)
		return await self._register_servers(
			server_map=server_map,
			tool_enhancements=tool_enhancements,
			clear_existing=clear_existing,
		)

	async def register_servers_from_addresses(
		self,
		addresses: List[str],
		tool_enhancements: Optional[Dict[str, Dict[str, Any]]] = None,
		clear_existing: bool = False,
	) -> List[BaseTool]:
		merged_server_map: Dict[str, Dict[str, Any]] = {}

		for address in addresses:
			json_text = self._read_json_by_address(address)
			config_data = self._parse_json_text(json_text)
			server_map = self._extract_server_map(config_data)
			for server_name, server_config in server_map.items():
				# 后出现的 server_name 覆盖之前配置，便于批量文件按优先级合并
				merged_server_map[server_name] = deepcopy(server_config)

		return await self._register_servers(
			server_map=merged_server_map,
			tool_enhancements=tool_enhancements,
			clear_existing=clear_existing,
		)

	async def _register_servers(
		self,
		server_map: Dict[str, Dict[str, Any]],
		tool_enhancements: Optional[Dict[str, Dict[str, Any]]],
		clear_existing: bool,
	) -> List[BaseTool]:
		if clear_existing:
			self.clear_tools()
			self.server_configs.clear()

		registered_tools: List[BaseTool] = []
		enhancements = tool_enhancements or {}

		for server_name, raw_server_config in server_map.items():
			normalized_config = self._normalize_server_config(server_name, raw_server_config)
			fetched_tools = await self.adapter.fetch_tools(server_name=server_name, server_config=normalized_config)

			enhanced_tools = [
				self._apply_enhancements(
					tool=tool,
					server_name=server_name,
					enhancements=enhancements,
				)
				for tool in fetched_tools
			]

			self.server_configs[server_name] = deepcopy(normalized_config)
			self._upsert_tools_by_name(enhanced_tools)
			registered_tools.extend(enhanced_tools)

		return registered_tools

	def enhance_tool_by_id(
		self,
		id: str,
		description: Optional[str] = None,
		labels: Optional[List[str]] = None,
		guard_policy: Optional[GuardPolicy] = None,
		ask_human_policy: Optional[AskHumanPolicy] = None,
	) -> bool:
		for index, tool in enumerate(self.tools):
			if tool.id == id:
				updated_tool = tool.model_copy(
					update={
						"description": description if description is not None else tool.description,
						"labels": list(labels) if labels is not None else tool.labels,
						"guard_policy": guard_policy if guard_policy is not None else tool.guard_policy,
						"ask_human_policy": ask_human_policy if ask_human_policy is not None else tool.ask_human_policy,
					}
				)
				self.tools[index] = updated_tool
				return True
		return False

	# === 增 ===
	def add_tool(self, tool: BaseTool) -> None:
		self.tools.append(tool)

	# === 删 ===
	def remove_tool_by_id(self, id: str) -> Optional[BaseTool]:
		for index, tool in enumerate(self.tools):
			if tool.id == id:
				return self.tools.pop(index)
		return None

	def clear_tools(self) -> None:
		self.tools.clear()

	# === 改 ===
	def update_tool_by_id(self, id: str, new_tool: BaseTool) -> bool:
		for index, tool in enumerate(self.tools):
			if tool.id == id:
				self.tools[index] = new_tool.model_copy(update={"id": id})
				return True
		return False

	def update_tools(self, new_tools: List[BaseTool]) -> None:
		self.tools = list(new_tools)

	# === 查 ===
	def get_tools(self) -> List[BaseTool]:
		return list(self.tools)

	def get_tool_count(self) -> int:
		return len(self.tools)

	def get_tool_by_id(self, id: str) -> Optional[BaseTool]:
		for tool in self.tools:
			if tool.id == id:
				return tool
		return None

	def get_server_configs(self) -> Dict[str, Dict[str, Any]]:
		return deepcopy(self.server_configs)

	def _upsert_tools_by_name(self, new_tools: List[BaseTool]) -> None:
		tools_by_name: Dict[str, BaseTool] = {tool.name: tool for tool in self.tools}
		for tool in new_tools:
			tools_by_name[tool.name] = tool
		self.tools = list(tools_by_name.values())

	def _parse_json_text(self, json_text: str) -> Dict[str, Any]:
		try:
			data = json.loads(json_text)
		except json.JSONDecodeError as e:
			raise ValueError(f"Invalid JSON config: {e}") from e

		if not isinstance(data, dict):
			raise ValueError("MCP config root must be a JSON object.")
		return data

	def _extract_server_map(self, config_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
		if "mcpServers" in config_data:
			servers = config_data.get("mcpServers")
			if not isinstance(servers, dict):
				raise ValueError("`mcpServers` must be a JSON object.")
			return self._validate_server_map(servers)

		# 兼容直接把 server map 作为根对象
		return self._validate_server_map(config_data)

	def _validate_server_map(self, server_map: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
		validated: Dict[str, Dict[str, Any]] = {}
		for server_name, server_config in server_map.items():
			if not isinstance(server_name, str) or not server_name.strip():
				raise ValueError("Server name must be a non-empty string.")
			if not isinstance(server_config, dict):
				raise ValueError(f"Server config for `{server_name}` must be a JSON object.")
			validated[server_name] = deepcopy(server_config)
		return validated

	def _normalize_server_config(self, server_name: str, server_config: Dict[str, Any]) -> Dict[str, Any]:
		normalized = deepcopy(server_config)

		has_command = "command" in normalized and bool(normalized.get("command"))
		has_url = "url" in normalized and bool(normalized.get("url"))
		if has_command and has_url:
			raise ValueError(f"Server `{server_name}` cannot contain both `command` and `url`.")
		if not has_command and not has_url:
			raise ValueError(f"Server `{server_name}` must contain either `command` or `url`.")

		if has_command:
			explicit_transport = normalized.get("transport")
			if explicit_transport is not None and str(explicit_transport).strip().lower() not in {"stdio"}:
				raise ValueError(f"Server `{server_name}` with `command` only supports transport `stdio`.")
			normalized["transport"] = "stdio"
			normalized["args"] = list(normalized.get("args", []))
			env = normalized.get("env")
			if env is not None and not isinstance(env, dict):
				raise ValueError(f"`env` for server `{server_name}` must be a JSON object.")

		if has_url:
			explicit_transport = normalized.get("transport")
			if explicit_transport is not None:
				transport = str(explicit_transport).strip().lower().replace("-", "").replace("_", "")
				if transport not in {"streamablehttp", "sse"}:
					raise ValueError(
						f"Server `{server_name}` with `url` only supports transport `sse` or `streamablehttp`."
					)
				normalized["transport"] = "streamablehttp" if transport == "streamablehttp" else "sse"
			else:
				normalized["transport"] = self._infer_url_transport(str(normalized["url"]))
			headers = normalized.get("headers")
			if headers is not None and not isinstance(headers, dict):
				raise ValueError(f"`headers` for server `{server_name}` must be a JSON object.")

		return normalized

	def _infer_url_transport(self, url: str) -> str:
		lower_url = url.lower()
		if "/sse" in lower_url:
			return "sse"
		return "streamablehttp"

	def _apply_enhancements(
		self,
		tool: BaseTool,
		server_name: str,
		enhancements: Dict[str, Dict[str, Any]],
	) -> BaseTool:
		merged: Dict[str, Any] = {}
		for key in self._build_enhancement_lookup_keys(tool=tool, server_name=server_name):
			value = enhancements.get(key)
			if isinstance(value, dict):
				merged.update(value)

		if not merged:
			return tool

		labels = tool.labels
		if "labels" in merged and merged["labels"] is not None:
			labels = list(merged["labels"])

		return tool.model_copy(
			update={
				"name": merged.get("name", tool.name),
				"description": merged.get("description", tool.description),
				"labels": labels,
				"guard_policy": merged.get("guard_policy", tool.guard_policy),
				"ask_human_policy": merged.get("ask_human_policy", tool.ask_human_policy),
			}
		)

	def _build_enhancement_lookup_keys(self, tool: BaseTool, server_name: str) -> List[str]:
		original_tool_name = self._get_original_tool_name(tool)
		keys = ["*"]

		# 通用 key：按工具名匹配
		keys.append(tool.name)
		if original_tool_name is not None:
			keys.append(original_tool_name)

		# 精确 key：按 server + 工具名匹配
		keys.append(f"{server_name}:{tool.name}")
		if original_tool_name is not None:
			keys.append(f"{server_name}:{original_tool_name}")

		return keys

	def _get_original_tool_name(self, tool: BaseTool) -> Optional[str]:
		for label in tool.labels or []:
			if label.startswith("mcp_original_tool:"):
				return label.split(":", 1)[1]
		return None

	def _read_json_by_address(self, address: str) -> str:
		if address.startswith("http://") or address.startswith("https://"):
			request = Request(address, headers={"User-Agent": "FastAgent-MCPManager"})
			with urlopen(request, timeout=10) as response:
				return response.read().decode("utf-8")

		path = Path(address).expanduser()
		if not path.exists() or not path.is_file():
			raise FileNotFoundError(f"JSON config file not found: {address}")
		return path.read_text(encoding="utf-8")

