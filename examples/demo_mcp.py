"""
MCP 工具调用演示（固定流程）

目标：
1. 通过 JSON 文件地址注入 MCP Server 配置（mcpServer.json）。
2. 使用 MCPManager 拉取并注册 MCP 工具。
3. 对一半工具注入 Guard 增强，调用前询问用户是否批准。
4. 运行固定的一次 Agent 流程，不做循环对话。
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fast_agent.resources.agent.agent import Agent
from fast_agent.resources.agent.event.events import (
	AssistantMessageChunkOutputEvent,
	AssistantMessageOutputEvent,
	GuardTriggeredEvent,
	RoundStopEvent,
	ToolCallEvent,
	ToolsExecutedEvent,
)
from fast_agent.resources.agent.lifespan.lifespan_manager import LifespanManager
from fast_agent.resources.context.context import Context
from fast_agent.resources.llm.llm_config import LLMConfig
from fast_agent.resources.mcp.mcp_manager import MCPManager
from fast_agent.types.messages.domain.tool_result_message import ToolResultMessage
from fast_agent.types.messages.domain.user_message import UserMessage
from fast_agent.types.tool.domain.guard_policy import GuardPolicy, GuardPolicyHumanResponseSchema


class Style:
	RESET = "\033[0m"
	BOLD = "\033[1m"
	GREEN = "\033[32m"
	YELLOW = "\033[33m"
	BLUE = "\033[34m"
	MAGENTA = "\033[35m"
	CYAN = "\033[36m"
	GRAY = "\033[90m"
	RED = "\033[31m"


class MCPToolApprovalResponse(GuardPolicyHumanResponseSchema):
	approved: bool = False
	reason: str = ""


def _guard_func(response: MCPToolApprovalResponse) -> bool:
	return getattr(response, "approved", False)


def _reject_func(response: MCPToolApprovalResponse) -> ToolResultMessage:
	return ToolResultMessage(
		tool_call_id="_placeholder_",
		name="mcp_tool",
		content=f"调用被用户拒绝，原因: {getattr(response, 'reason', '用户未批准')}",
		is_error=False,
	)


def print_header(title: str):
	print(f"\n{Style.CYAN}{Style.BOLD}{'=' * 70}{Style.RESET}")
	print(f"{Style.CYAN}{Style.BOLD}{title}{Style.RESET}")
	print(f"{Style.CYAN}{Style.BOLD}{'=' * 70}{Style.RESET}")


def print_info(msg: str):
	print(f"{Style.BLUE}INFO{Style.RESET} {msg}")


def print_ok(msg: str):
	print(f"{Style.GREEN}OK{Style.RESET} {msg}")


def print_warn(msg: str):
	print(f"{Style.YELLOW}WARN{Style.RESET} {msg}")


def print_err(msg: str):
	print(f"{Style.RED}ERR{Style.RESET} {msg}")


async def handle_guard(agent: Agent, event: GuardTriggeredEvent):
	contexts = event.data.guard_triggered_contexts
	snapshot = event.data.snapshot

	print_header("Guard 审批")
	print_warn(f"检测到 {len(contexts)} 个敏感工具调用，需人工审批")

	human_response: Dict[str, GuardPolicyHumanResponseSchema] = {}
	for ctx in contexts:
		tool_call = ctx.tool_call
		print(f"\n{Style.MAGENTA}工具: {tool_call.function_name}{Style.RESET}")
		print(f"{Style.GRAY}参数: {tool_call.function_args}{Style.RESET}")
		user_text = await asyncio.get_event_loop().run_in_executor(
			None,
			lambda: input(f"{Style.GREEN}是否批准调用? (y/n): {Style.RESET}"),
		)
		approved = user_text.strip().lower() in ("y", "yes", "是")
		reason = ""
		if not approved:
			reason = await asyncio.get_event_loop().run_in_executor(
				None,
				lambda: input(f"{Style.GRAY}拒绝原因(可选): {Style.RESET}"),
			)
		human_response[tool_call.tool_call_id] = MCPToolApprovalResponse(
			approved=approved,
			reason=reason,
		)

	print_info("使用 snapshot 恢复执行")
	await process_resume_stream(agent, snapshot, human_response)


async def process_stream(agent: Agent, user_message: UserMessage):
	reasoning_started = False
	content_started = False

	async for event in agent.stream(user_message, stream_mode="chunk"):
		if isinstance(event, AssistantMessageChunkOutputEvent):
			data = event.data
			if data.chunk_type == "reasoning_content" and data.reasoning_content:
				if not reasoning_started:
					reasoning_started = True
					print(f"\n{Style.MAGENTA}思考: {Style.RESET}", end="")
				print(data.reasoning_content, end="", flush=True)

			if data.chunk_type == "content" and data.content:
				if reasoning_started and not content_started:
					print()
					reasoning_started = False
				if not content_started:
					content_started = True
					print(f"\n{Style.GREEN}助手: {Style.RESET}", end="")
				print(data.content, end="", flush=True)

		elif isinstance(event, AssistantMessageOutputEvent):
			if content_started or reasoning_started:
				print()
			if not content_started and event.data.content:
				print(f"\n{Style.GREEN}助手: {event.data.content}{Style.RESET}")

		elif isinstance(event, ToolCallEvent):
			print(f"\n{Style.YELLOW}调用工具: {event.data.function_name}{Style.RESET}")
			print(f"{Style.GRAY}参数: {event.data.function_args}{Style.RESET}")

		elif isinstance(event, ToolsExecutedEvent):
			print(f"\n{Style.YELLOW}工具执行结果:{Style.RESET}")
			for r in event.data.tool_results:
				print(f"- {r.name}: {r.content}")

		elif isinstance(event, GuardTriggeredEvent):
			print()
			await handle_guard(agent, event)
			return

		elif isinstance(event, RoundStopEvent):
			print(f"\n{Style.CYAN}流程完成，轮次结束。{Style.RESET}")


async def process_resume_stream(
	agent: Agent,
	snapshot,
	human_response: Dict[str, GuardPolicyHumanResponseSchema],
):
	async for event in agent.resume_stream(snapshot, human_response, stream_mode="chunk"):
		if isinstance(event, ToolCallEvent):
			print(f"\n{Style.YELLOW}[恢复] 调用工具: {event.data.function_name}{Style.RESET}")
		elif isinstance(event, ToolsExecutedEvent):
			print(f"\n{Style.YELLOW}[恢复] 工具执行结果:{Style.RESET}")
			for r in event.data.tool_results:
				print(f"- {r.name}: {r.content}")
		elif isinstance(event, AssistantMessageOutputEvent) and event.data.content:
			print(f"\n{Style.GREEN}[恢复] 助手: {event.data.content}{Style.RESET}")
		elif isinstance(event, RoundStopEvent):
			print(f"\n{Style.CYAN}恢复流程完成。{Style.RESET}")


async def main():
	load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

	print_header("MCP 固定流程演示")

	api_key = os.environ.get("DEEPSEEK_API_KEY", "")
	if not api_key:
		print_err("缺少 DEEPSEEK_API_KEY，请先在环境变量或 examples/.env 中设置。")
		return

	llm_config = LLMConfig(
		model_name=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
		api_key=api_key,
		base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
		provider="deepseek",
		temperature=0.2,
		top_p=0.8,
		max_tokens=2048,
	)

	manager = MCPManager()

	repo_root = Path(__file__).resolve().parent.parent
	mcp_config_path = repo_root / "static" / "json" / "mcpServer.json"
	if not mcp_config_path.exists():
		print_err(f"未找到 MCP 配置文件: {mcp_config_path}")
		return

	print_info(f"从 JSON 地址注册 MCP 工具: {mcp_config_path}")
	registered_tools = await manager.register_servers_from_addresses([str(mcp_config_path)])
	if not registered_tools:
		print_err("未从 MCP Server 拉取到任何工具。")
		return

	print_ok(f"成功注册 {len(registered_tools)} 个 MCP 工具")
	for t in registered_tools:
		print(f"- {t.name}")

	guarded_count = len(registered_tools) // 2
	if len(registered_tools) == 1:
		guarded_count = 1

	for tool in registered_tools[:guarded_count]:
		manager.enhance_tool_by_id(
			id=tool.id,
			guard_policy=GuardPolicy(
				info="MCP 工具调用需要人工确认",
				schema=MCPToolApprovalResponse(),
				guard_func=_guard_func,
				reject_func=_reject_func,
			),
			labels=list((tool.labels or []) + ["guarded"]),
			description=f"[Guard增强] {tool.description or ''}".strip(),
		)

	print_ok(f"已对 {guarded_count}/{len(registered_tools)} 个工具应用 Guard 增强")

	context = Context(
		system_prompt=(
			"你是 MCP 工具测试助手。"
			"请严格按需调用工具完成任务。"
			"如果用户让你联网检索，请优先使用 MCP web search 工具。"
			"最终用中文给出简洁结论，并附关键来源信息。"
		),
		tool_manager=manager.get_tools(),
	)

	agent = Agent(
		llm_config=llm_config,
		context=context,
		lifespan_manager=LifespanManager(),
	)

	fixed_user_text = (
		"请使用可用工具联网检索：总结今天 AI Agent 领域 3 条重要动态，"
		"每条给出一句摘要，并标注来源。"
	)
	print_header("固定流程开始")
	print_info(f"固定用户输入: {fixed_user_text}")

	await process_stream(agent, UserMessage(content=fixed_user_text))
	print_header("Demo 结束")


if __name__ == "__main__":
	asyncio.run(main())

