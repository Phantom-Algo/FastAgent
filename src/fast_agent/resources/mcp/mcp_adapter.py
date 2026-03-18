from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.types import (
	AudioContent,
	BlobResourceContents,
	CallToolResult,
	EmbeddedResource,
	ImageContent,
	ResourceLink,
	TextContent,
	TextResourceContents,
)
import httpx
from pydantic import BaseModel, ConfigDict, create_model

from ...types.messages.domain import BasePart, ImagePart, TextPart
from ...types.mcp.base_mcp_adapter import BaseMCPAdapter
from ...types.tool.base_tool import BaseTool
from ..tool.tool import Tool


class MCPAdapter(BaseMCPAdapter):
	"""将 MCP Server 工具适配为框架 BaseTool 的实现。"""

	def __init__(self, prefix_server_name: bool = True):
		self.prefix_server_name = prefix_server_name

	async def fetch_tools(self, server_name: str, server_config: Dict[str, Any]) -> List[BaseTool]:
		transport = self._resolve_transport(server_config)
		if transport == "stdio":
			return await self.fetch_tools_via_stdio(server_name=server_name, server_config=server_config)
		if transport == "sse":
			return await self.fetch_tools_via_sse(server_name=server_name, server_config=server_config)
		if transport == "streamablehttp":
			return await self.fetch_tools_via_streamablehttp(server_name=server_name, server_config=server_config)
		raise ValueError(f"Unsupported transport type: {transport}")

	async def fetch_tools_via_stdio(self, server_name: str, server_config: Dict[str, Any]) -> List[BaseTool]:
		required_key = "command"
		if required_key not in server_config or not server_config.get(required_key):
			raise ValueError(f"stdio config for server `{server_name}` must include non-empty `command`.")

		async with self._open_session(server_name=server_name, server_config=server_config) as session:
			tools_result = await session.list_tools()
			return [
				self._convert_mcp_tool_to_base_tool(
					server_name=server_name,
					server_config=server_config,
					mcp_tool=tool,
				)
				for tool in tools_result.tools
			]

	async def fetch_tools_via_streamablehttp(self, server_name: str, server_config: Dict[str, Any]) -> List[BaseTool]:
		required_key = "url"
		if required_key not in server_config or not server_config.get(required_key):
			raise ValueError(f"streamablehttp config for server `{server_name}` must include non-empty `url`.")

		async with self._open_session(server_name=server_name, server_config=server_config) as session:
			tools_result = await session.list_tools()
			return [
				self._convert_mcp_tool_to_base_tool(
					server_name=server_name,
					server_config=server_config,
					mcp_tool=tool,
				)
				for tool in tools_result.tools
			]

	async def fetch_tools_via_sse(self, server_name: str, server_config: Dict[str, Any]) -> List[BaseTool]:
		required_key = "url"
		if required_key not in server_config or not server_config.get(required_key):
			raise ValueError(f"sse config for server `{server_name}` must include non-empty `url`.")

		async with self._open_session(server_name=server_name, server_config=server_config) as session:
			tools_result = await session.list_tools()
			return [
				self._convert_mcp_tool_to_base_tool(
					server_name=server_name,
					server_config=server_config,
					mcp_tool=tool,
				)
				for tool in tools_result.tools
			]

	@asynccontextmanager
	async def _open_session(self, server_name: str, server_config: Dict[str, Any]):
		transport = self._resolve_transport(server_config)

		if transport == "stdio":
			stdio_params = StdioServerParameters(
				command=server_config["command"],
				args=list(server_config.get("args", [])),
				env=server_config.get("env"),
				cwd=server_config.get("cwd"),
			)
			async with stdio_client(stdio_params) as (read_stream, write_stream):
				async with ClientSession(read_stream=read_stream, write_stream=write_stream) as session:
					await session.initialize()
					yield session
			return

		if transport == "streamablehttp":
			timeout_value = server_config.get("timeout", 30)
			sse_read_timeout_value = server_config.get("sse_read_timeout", 300)
			headers = server_config.get("headers")

			timeout_seconds = timeout_value.total_seconds() if isinstance(timeout_value, timedelta) else float(timeout_value)
			sse_read_timeout_seconds = (
				sse_read_timeout_value.total_seconds()
				if isinstance(sse_read_timeout_value, timedelta)
				else float(sse_read_timeout_value)
			)

			http_client = create_mcp_http_client(
				headers=headers,
				timeout=httpx.Timeout(timeout_seconds, read=sse_read_timeout_seconds),
			)

			async with http_client:
				async with streamable_http_client(
					url=server_config["url"],
					http_client=http_client,
				) as (read_stream, write_stream, _):
					async with ClientSession(read_stream=read_stream, write_stream=write_stream) as session:
						await session.initialize()
						yield session
			return

		if transport == "sse":
			async with sse_client(
				url=server_config["url"],
				headers=server_config.get("headers"),
				timeout=server_config.get("timeout", 5),
				sse_read_timeout=server_config.get("sse_read_timeout", 300),
			) as (read_stream, write_stream):
				async with ClientSession(read_stream=read_stream, write_stream=write_stream) as session:
					await session.initialize()
					yield session
			return

		raise ValueError(f"Unsupported transport `{transport}` for server `{server_name}`.")

	def _convert_mcp_tool_to_base_tool(
		self,
		server_name: str,
		server_config: Dict[str, Any],
		mcp_tool: Any,
	) -> BaseTool:
		original_tool_name = mcp_tool.name
		final_tool_name = self._build_tool_name(server_name=server_name, original_name=original_tool_name)

		input_schema = mcp_tool.inputSchema if getattr(mcp_tool, "inputSchema", None) else {"type": "object", "properties": {}}
		args_schema = self._build_args_model_from_json_schema(
			schema=input_schema,
			model_name=f"MCP{self._sanitize_model_name(server_name)}{self._sanitize_model_name(original_tool_name)}Args",
		)

		async def _mcp_tool_func(**kwargs):
			async with self._open_session(server_name=server_name, server_config=server_config) as session:
				result: CallToolResult = await session.call_tool(name=original_tool_name, arguments=kwargs)
			if result.isError:
				raise RuntimeError(f"MCP tool `{server_name}:{original_tool_name}` returned error result: {self._to_plain_data(result)}")
			return self._format_call_tool_result(result)

		description = mcp_tool.description if getattr(mcp_tool, "description", None) else ""
		labels = ["mcp", f"mcp_server:{server_name}", f"mcp_original_tool:{original_tool_name}"]

		return Tool(
			name=final_tool_name,
			description=description,
			args_schema=args_schema,
			func=_mcp_tool_func,
			is_async=True,
			labels=labels,
		)

	def _build_tool_name(self, server_name: str, original_name: str) -> str:
		if not self.prefix_server_name:
			return original_name
		return f"{server_name}__{original_name}"

	def _build_args_model_from_json_schema(self, schema: Dict[str, Any], model_name: str) -> Type[BaseModel]:
		normalized_schema = schema if isinstance(schema, dict) else {"type": "object", "properties": {}}
		if normalized_schema.get("type") != "object":
			normalized_schema = {
				"type": "object",
				"properties": {"value": normalized_schema},
				"required": ["value"],
			}

		properties = normalized_schema.get("properties", {})
		required_fields = set(normalized_schema.get("required", []))
		allow_extra = normalized_schema.get("additionalProperties", True)

		fields: Dict[str, Tuple[Any, Any]] = {}
		for field_name, field_schema in properties.items():
			annotation = self._json_schema_to_python_type(
				schema=field_schema,
				fallback_model_name=f"{model_name}{self._sanitize_model_name(field_name)}",
			)
			default = ... if field_name in required_fields else None
			fields[field_name] = (annotation, default)

		config = ConfigDict(extra="allow" if allow_extra else "forbid")
		return create_model(model_name, __config__=config, **fields)

	def _json_schema_to_python_type(self, schema: Any, fallback_model_name: str) -> Any:
		if not isinstance(schema, dict):
			return Any

		enum_values = schema.get("enum")
		if isinstance(enum_values, list) and enum_values:
			unique_enum = tuple(dict.fromkeys(enum_values))
			from typing import Literal

			return Literal.__getitem__(unique_enum)

		if "anyOf" in schema and isinstance(schema["anyOf"], list) and schema["anyOf"]:
			return Union.__getitem__(tuple(self._json_schema_to_python_type(s, fallback_model_name) for s in schema["anyOf"]))

		if "oneOf" in schema and isinstance(schema["oneOf"], list) and schema["oneOf"]:
			return Union.__getitem__(tuple(self._json_schema_to_python_type(s, fallback_model_name) for s in schema["oneOf"]))

		schema_type = schema.get("type")

		if isinstance(schema_type, list):
			has_null = "null" in schema_type
			non_null = [t for t in schema_type if t != "null"]
			if not non_null:
				return Any
			if len(non_null) == 1:
				py_type = self._json_schema_to_python_type({**schema, "type": non_null[0]}, fallback_model_name)
			else:
				py_type = Union.__getitem__(
					tuple(self._json_schema_to_python_type({**schema, "type": t}, fallback_model_name) for t in non_null)
				)
			return Optional[py_type] if has_null else py_type

		if schema_type == "string":
			return str
		if schema_type == "integer":
			return int
		if schema_type == "number":
			return float
		if schema_type == "boolean":
			return bool
		if schema_type == "array":
			items_schema = schema.get("items", {})
			item_type = self._json_schema_to_python_type(items_schema, fallback_model_name=f"{fallback_model_name}Item")
			return List[item_type]
		if schema_type == "object" or "properties" in schema:
			nested_name = self._sanitize_model_name(fallback_model_name)
			return self._build_args_model_from_json_schema(schema=schema, model_name=nested_name)

		return Any

	def _resolve_transport(self, server_config: Dict[str, Any]) -> str:
		if "transport" in server_config and server_config["transport"]:
			transport = str(server_config["transport"]).strip().lower()
			if transport in {"streamable-http", "streamable_http"}:
				return "streamablehttp"
			return transport
		if "command" in server_config:
			return "stdio"
		if "url" in server_config:
			url = str(server_config["url"]).lower()
			if "/sse" in url:
				return "sse"
			return "streamablehttp"
		raise ValueError("MCP server config must contain either `command` (stdio) or `url` (sse/streamablehttp).")

	def _format_call_tool_result(self, result: CallToolResult) -> Union[str, List[BasePart]]:
		parts: List[BasePart] = []

		for block in result.content:
			# --- 文本数据 ---
			if isinstance(block, TextContent):
				parts.append(TextPart(text=block.text))
				continue

			# --- 图片数据 ---
			if isinstance(block, ImageContent):
				parts.append(ImagePart(base64_data=block.data, mime_type=block.mimeType))
				continue

			# --- 资源链接 ---
			if isinstance(block, ResourceLink):
				# 若为图片资源，则允许将图片URL作为图片输入
				if block.mimeType and block.mimeType.startswith("image/"):
					parts.append(ImagePart(url=block.uri))

				# 否则将链接信息作为文本输入，确保内容不丢失
				else:
					link_payload = {
						"name": block.name,
						"uri": block.uri,
						"mimeType": block.mimeType,
						"description": block.description,
					}
					parts.append(TextPart(text=f"ResourceLink: {json.dumps(link_payload, ensure_ascii=False)}"))
				continue

			if isinstance(block, EmbeddedResource):
				# TODO：向量数据待实现
				# resource = block.resource

				# if isinstance(resource, TextResourceContents):
				# 	parts.append(TextPart(text=resource.text))
				# 	continue

				# if isinstance(resource, BlobResourceContents):
				# 	mime_type = resource.mimeType or "application/octet-stream"
				# 	if mime_type.startswith("image/"):
				# 		parts.append(ImagePart(base64_data=resource.blob, mime_type=mime_type))
				# 	else:
				# 		blob_payload = {
				# 			"uri": resource.uri,
				# 			"mimeType": mime_type,
				# 			"blob_base64": resource.blob,
				# 		}
				# 		parts.append(TextPart(text=f"EmbeddedResource(blob): {json.dumps(blob_payload, ensure_ascii=False)}"))
				# 	continue

				# embedded_payload = self._to_plain_data(resource)
				# parts.append(TextPart(text=f"EmbeddedResource(unknown): {json.dumps(embedded_payload, ensure_ascii=False, default=str)}"))
				continue

			if isinstance(block, AudioContent):
				# TODO：音频内容待实现
				# audio_payload = {
				# 	"mimeType": block.mimeType,
				# 	"audio_base64": block.data,
				# }
				# parts.append(TextPart(text=f"AudioContent: {json.dumps(audio_payload, ensure_ascii=False)}"))
				continue

		if result.structuredContent is not None:
			structured_payload = self._to_plain_data(result.structuredContent)
			parts.insert(0, TextPart(text=f"structured_content: {json.dumps(structured_payload, ensure_ascii=False, default=str)}"))

		if parts:
			return parts

		# 若无内容块但有结构化内容，则返回结构化内容的 JSON 字符串，确保结果不丢失
		if result.structuredContent is not None:
			return json.dumps(self._to_plain_data(result.structuredContent), ensure_ascii=False, default=str)

		return ""

	def _to_plain_data(self, value: Any) -> Any:
		if isinstance(value, BaseModel):
			return {k: self._to_plain_data(v) for k, v in value.model_dump(mode="python").items()}
		if isinstance(value, dict):
			return {k: self._to_plain_data(v) for k, v in value.items()}
		if isinstance(value, list):
			return [self._to_plain_data(v) for v in value]
		if isinstance(value, tuple):
			return [self._to_plain_data(v) for v in value]
		return value

	def _sanitize_model_name(self, text: str) -> str:
		cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", text)
		parts = [p for p in cleaned.split("_") if p]
		if not parts:
			return "Model"
		return "".join(p[:1].upper() + p[1:] for p in parts)

