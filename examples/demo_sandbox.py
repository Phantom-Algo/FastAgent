from __future__ import annotations

import asyncio
import os
import sys
from datetime import timedelta

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fast_agent.resources.agent.agent import Agent
from fast_agent.resources.agent.event.events import (
	AssistantMessageChunkOutputEvent,
	AssistantMessageOutputEvent,
	RoundStopEvent,
	ToolCallEvent,
	ToolsExecutedEvent,
)
from fast_agent.resources.context.context import Context
from fast_agent.resources.llm.llm_config import LLMConfig
from fast_agent.resources.sandbox.factory.opensandbox_factory import OpenSandboxFactory
from fast_agent.resources.tool.tool_creator import tool_creator
from fast_agent.types.messages.domain.user_message import UserMessage
from fast_agent.types.sandbox.base_sandbox import ISandBox
from fast_agent.types.sandbox.domain.command_options import CommandOpts


DANGEROUS_COMMAND_REGISTRY: dict[str, str] = {
	"rm -rf": "Recursive deletion can destroy files permanently.",
	"mkfs": "Formatting disks can destroy all data.",
	"shutdown": "Shutdown can interrupt current work.",
	"reboot": "Reboot can interrupt current work.",
	"dd if=": "Raw disk write may corrupt system data.",
	"> /": "Root-level redirection may overwrite critical files.",
}


SANDBOX_CLIENT: ISandBox | None = None


def _detect_risk(command: str) -> str | None:
	normalized = command.lower().strip()
	for keyword, reason in DANGEROUS_COMMAND_REGISTRY.items():
		if keyword in normalized:
			return f"Matched keyword '{keyword}'. {reason}"
	return None


async def _ask_user_approval(command: str, risk_reason: str) -> bool:
	print("\n[HITL] risky command detected")
	print(f"  command: {command}")
	print(f"  reason : {risk_reason}")
	loop = asyncio.get_event_loop()
	answer = await loop.run_in_executor(None, lambda: input("Approve execution? (y/n): "))
	return answer.strip().lower() in ("y", "yes")


def _format_execution_result(command: str, result) -> str:
	lines: list[str] = [f"command: {command}"]
	if result.id:
		lines.append(f"execution_id: {result.id}")
	lines.append(f"ok: {result.ok}")

	if result.logs.stdout:
		lines.append("stdout:")
		lines.extend(msg.text for msg in result.logs.stdout)

	if result.logs.stderr:
		lines.append("stderr:")
		lines.extend(msg.text for msg in result.logs.stderr)

	if result.result:
		lines.append("result:")
		lines.extend(item.text for item in result.result if item.text)

	if result.error is not None:
		lines.append(f"error: {result.error.name}: {result.error.value}")
		if result.error.traceback:
			lines.append("traceback:")
			lines.extend(result.error.traceback)

	return "\n".join(lines)


@tool_creator(
	tool_name="sandbox_command",
	tool_description=(
		"Execute one shell command in sandbox. "
		"If command is risky, tool asks user for short-term approval before execution."
	),
	labels=["sandbox", "shell", "hitl"],
)
async def sandbox_command(command: str) -> str:
	if SANDBOX_CLIENT is None:
		return "sandbox is not initialized"

	risk = _detect_risk(command)
	if risk is not None:
		approved = await _ask_user_approval(command=command, risk_reason=risk)
		if not approved:
			return f"blocked by user review: {risk}"

	result = await SANDBOX_CLIENT.command(
		command,
		opts=CommandOpts(timeout=timedelta(seconds=45)),
	)
	return _format_execution_result(command=command, result=result)


async def _stream_once(agent: Agent, user_text: str) -> None:
	reasoning_open = False
	content_open = False

	async for event in agent.stream(UserMessage(content=user_text), stream_mode="chunk"):
		if isinstance(event, AssistantMessageChunkOutputEvent):
			data = event.data

			if data.chunk_type == "reasoning_content" and data.reasoning_content:
				if not reasoning_open:
					reasoning_open = True
					print("\n[thinking] ", end="", flush=True)
				print(data.reasoning_content, end="", flush=True)

			elif data.chunk_type == "content" and data.content:
				if reasoning_open and not content_open:
					print()
					reasoning_open = False
				if not content_open:
					content_open = True
					print("\nassistant: ", end="", flush=True)
				print(data.content, end="", flush=True)

		elif isinstance(event, AssistantMessageOutputEvent):
			if content_open or reasoning_open:
				print()
			if not content_open and event.data.content:
				print(f"\nassistant: {event.data.content}")

		elif isinstance(event, ToolCallEvent):
			if content_open or reasoning_open:
				print()
			content_open = False
			reasoning_open = False
			print(f"\n[tool-call] {event.data.function_name} args={event.data.function_args}")

		elif isinstance(event, ToolsExecutedEvent):
			print("[tool-result]")
			for item in event.data.tool_results:
				print(f"- {item.name}: {item.content}")

		elif isinstance(event, RoundStopEvent):
			pass

	if content_open or reasoning_open:
		print()


async def _build_sandbox() -> ISandBox:
	factory = OpenSandboxFactory()

	domain = os.getenv("OPENSANDBOX_DOMAIN", "localhost:8080")
	api_key = os.getenv("OPENSANDBOX_API_KEY")
	image = os.getenv("OPENSANDBOX_IMAGE", "ubuntu:22.04")

	return await factory.create_sandbox(
		image=image,
		domain=domain,
		api_key=api_key,
		request_timeout=timedelta(seconds=60),
		timeout=timedelta(minutes=10),
	)


def _build_agent() -> Agent:
	api_key = os.getenv("DOUBAO_SEED_API_KEY", "")
	if not api_key:
		raise ValueError("missing DOUBAO_SEED_API_KEY")

	llm_config = LLMConfig(
		model_name=os.getenv("DOUBAO_SEED_MODEL", "doubao-seed-2-0-pro-260215"),
		api_key=api_key,
		base_url=os.getenv("DOUBAO_SEED_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
		provider="doubao_seed",
		temperature=0.1,
		top_p=0.9,
		max_tokens=2048,
	)

	context = Context(
		system_prompt=(
			"你是一个智能助手，拥有自己的一个独立沙箱环境，可以执行用户授权的shell命令来获取信息或完成任务。 "
		),
		tool_manager=[sandbox_command],
	)

	return Agent(llm_config=llm_config, context=context)


async def main() -> None:
	global SANDBOX_CLIENT

	load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

	try:
		SANDBOX_CLIENT = await _build_sandbox()
	except Exception as e:
		print("sandbox init failed")
		print(f"reason: {type(e).__name__}: {e}")
		print("hint: set OPENSANDBOX_DOMAIN/SANDBOX_DOMAIN to a real OpenSandbox lifecycle endpoint")
		print("hint: if server requires API key, set OPENSANDBOX_API_KEY or SANDBOX_API_KEY")
		return

	agent = _build_agent()

	print("sandbox demo started")
	print("type quit or exit to stop")

	try:
		while True:
			user_text = await asyncio.get_event_loop().run_in_executor(None, lambda: input("you: "))
			user_text = user_text.strip()
			if not user_text:
				continue
			if user_text.lower() in ("quit", "exit"):
				break

			await _stream_once(agent, user_text)
			print("-" * 60)
	finally:
		if SANDBOX_CLIENT is not None:
			try:
				await SANDBOX_CLIENT.kill()
			finally:
				await SANDBOX_CLIENT.close()
		print("sandbox demo stopped")


if __name__ == "__main__":
	asyncio.run(main())
