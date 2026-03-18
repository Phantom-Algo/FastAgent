"""
FastAgent 图像识别综合演示（豆包 Seed）

测试目标：
1. 用户输入多模态图片（本地 base64 + 公网 URL）
2. 工具调用返回多模态图片（ToolResultMessage.content = List[BasePart]）
3. 多轮复杂对话，验证 reasoning_content 与 tool_calls 混合场景稳定性
"""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

from fast_agent.resources.agent.agent import Agent
from fast_agent.resources.context.context import Context
from fast_agent.resources.llm.llm_config import LLMConfig
from fast_agent.resources.tool.tool_creator import tool_creator
from fast_agent.resources.agent.event.events import (
	AssistantMessageChunkOutputEvent,
	AssistantMessageOutputEvent,
	InterruptEvent,
	RoundStopEvent,
	ToolCallEvent,
	ToolsExecutedEvent,
)
from fast_agent.types.messages.domain import UserMessage
from fast_agent.types.messages.domain.content_part import BasePart, ImagePart, TextPart


PUBLIC_IMAGE_URL = "https://th.bing.com/th/id/OIP.qOVNXup8kQ0DZg7RTVN3ZwHaE8?w=252&h=180&c=7&r=0&o=7&dpr=2&pid=1.7&rm=3"
STATIC_IMG_DIR = Path(__file__).resolve().parent.parent / "static" / "img"
SUPPORTED_REMOTE_IMAGE_MIME_TYPES = {"image/jpeg", "image/png"}


def _guess_mime_type(file_path: Path) -> str:
	mime_type, _ = mimetypes.guess_type(str(file_path))
	return mime_type or "application/octet-stream"


def _load_local_image_part(file_name: str, detail: str = "high") -> ImagePart:
	file_path = STATIC_IMG_DIR / file_name
	if not file_path.exists():
		raise FileNotFoundError(f"Local image not found: {file_path}")

	raw = file_path.read_bytes()
	encoded = base64.b64encode(raw).decode("utf-8")
	return ImagePart(
		base64_data=encoded,
		mime_type=_guess_mime_type(file_path),
		detail=detail,
	)


def _download_public_image_as_base64(url: str) -> Tuple[Optional[ImagePart], str]:
	"""
	尝试把公网 URL 下载并转为 base64 图片。

	返回值：
	- (ImagePart, "ok")：可直接用于模型
	- (None, reason)：当前 URL 在环境下不可用或格式不受支持
	"""
	request = urllib.request.Request(
		url,
		headers={
			"User-Agent": "Mozilla/5.0",
			"Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
		},
	)

	try:
		with urllib.request.urlopen(request, timeout=20) as response:
			final_url = response.geturl()
			content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
			payload = response.read()
	except Exception as exc:
		return None, f"download_failed:{type(exc).__name__}:{str(exc)}"

	if not content_type.startswith("image/"):
		return None, f"not_image_content_type:{content_type or 'unknown'}:resolved_url={final_url}"

	if content_type not in SUPPORTED_REMOTE_IMAGE_MIME_TYPES:
		return None, f"unsupported_remote_mime:{content_type}:resolved_url={final_url}"

	encoded = base64.b64encode(payload).decode("utf-8")
	return ImagePart(base64_data=encoded, mime_type=content_type, detail="high"), "ok"


def _build_public_image_parts_for_prompt() -> Tuple[List[BasePart], str]:
	"""
	构造用于消息中的“公网图像测试”片段。

	优先：把指定 URL 解析为 base64 图片后送入模型。
	降级：若 URL 在当前环境不可直接得到图片，则回退本地图片，
	并保留 URL 与失败原因文本，确保“URL 路径已被测试”。
	"""
	parts: List[BasePart] = [TextPart(text=f"公网图片原始链接（按要求测试）: {PUBLIC_IMAGE_URL}")]

	public_image_part, status = _download_public_image_as_base64(PUBLIC_IMAGE_URL)
	if public_image_part is not None:
		parts.append(public_image_part)
		parts.append(TextPart(text="公网链接已成功下载并转为 base64 图片输入。"))
		return parts, "ok"

	parts.append(TextPart(text=f"公网链接图像直连失败，原因: {status}"))
	parts.append(TextPart(text="已自动回退为本地图片输入，避免模型请求中断。"))
	parts.append(_load_local_image_part("Web.png", detail="high"))
	return parts, status


@tool_creator(
	tool_name="get_reference_images",
	strict_mode=False,
	tool_description=(
		"返回用于视觉比对的参考素材，包含说明文本、本地图片和公网图片。"
		"当用户请求比对、验证、交叉检查或溯源时优先调用。"
	),
	labels=["vision", "multimodal", "reference"],
)
async def get_reference_images(scene: str = "generic", include_local: bool = True, include_public: bool = True) -> List[BasePart]:
	parts: List[BasePart] = [
		TextPart(
			text=(
				f"工具参考素材（scene={scene}）。请结合这些参考图与用户输入图做交叉验证，"
				"输出可核验结论并明确不确定项。"
			)
		)
	]

	if include_local:
		file_name = "Web.png" if "web" in scene.lower() else "Agent架构设计.png"
		try:
			parts.append(_load_local_image_part(file_name=file_name, detail="high"))
			parts.append(TextPart(text=f"本地参考图文件名: {file_name}"))
		except FileNotFoundError as err:
			parts.append(TextPart(text=f"本地参考图加载失败: {str(err)}"))

	if include_public:
		public_parts, status = _build_public_image_parts_for_prompt()
		parts.extend(public_parts)
		if status != "ok":
			parts.append(TextPart(text="提示：当前环境下该公网 URL 未返回可用图片格式，工具已自动降级处理。"))

	return parts


def _print_header(title: str) -> None:
	print("\n" + "=" * 72)
	print(title)
	print("=" * 72)


def _print_tool_results(results: List[Any]) -> None:
	for result in results:
		content = getattr(result, "content", "")
		if isinstance(content, str):
			print(f"[TOOL RESULT] {result.name}: {content}")
			continue

		if isinstance(content, list):
			part_types = [getattr(part, "type", "unknown") for part in content]
			print(f"[TOOL RESULT] {result.name}: multimodal parts={part_types}")
			continue

		print(f"[TOOL RESULT] {result.name}: {json.dumps(content, ensure_ascii=False, default=str)}")


async def _run_one_round(agent: Agent, user_message: UserMessage, round_name: str) -> None:
	_print_header(round_name)

	reasoning_started = False
	content_started = False

	async for event in agent.stream(user_message, stream_mode="chunk"):
		if isinstance(event, AssistantMessageChunkOutputEvent):
			chunk_data = event.data
			if chunk_data.chunk_type == "reasoning_content" and chunk_data.reasoning_content:
				if not reasoning_started:
					reasoning_started = True
					print("\n[REASONING] ", end="", flush=True)
				print(chunk_data.reasoning_content, end="", flush=True)

			elif chunk_data.chunk_type == "content" and chunk_data.content:
				if reasoning_started and not content_started:
					print()
					reasoning_started = False
				if not content_started:
					content_started = True
					print("\n[ASSISTANT] ", end="", flush=True)
				print(chunk_data.content, end="", flush=True)

		elif isinstance(event, ToolCallEvent):
			if reasoning_started or content_started:
				print()
				reasoning_started = False
				content_started = False
			print(f"\n[TOOL CALL] {event.data.function_name}: {json.dumps(event.data.function_args, ensure_ascii=False)}")

		elif isinstance(event, ToolsExecutedEvent):
			_print_tool_results(event.data.tool_results)

		elif isinstance(event, AssistantMessageOutputEvent):
			if reasoning_started or content_started:
				print()
				reasoning_started = False
				content_started = False

			if event.data.content:
				print(f"[ASSISTANT FINAL] {event.data.content}")

			if event.data.finish_reason:
				print(f"[FINISH REASON] {event.data.finish_reason}")

		elif isinstance(event, InterruptEvent):
			print(f"\n[INTERRUPT] {event.data.reason}")
			return

		elif isinstance(event, RoundStopEvent):
			pass


def _build_round_messages() -> List[tuple[str, UserMessage]]:
	round_messages: List[tuple[str, UserMessage]] = []
	public_round_parts, public_status = _build_public_image_parts_for_prompt()

	# Round 1: 用户输入本地图，要求调用工具返回参考图。
	round_messages.append(
		(
			"Round 1 - 本地图输入 + 工具参考图",
			UserMessage(
				content=[
					TextPart(
						text=(
							"这是我上传的本地架构图。请先调用 get_reference_images(scene='architecture-audit')，"
							"然后基于我的图 + 工具返回图做结构识别，并给出3条可验证结论。"
						)
					),
					_load_local_image_part("Agent架构设计.png", detail="high"),
				]
			),
		)
	)

	# Round 2: 用户输入公网 URL 图，继续要求工具交叉比对。
	round_messages.append(
		(
			"Round 2 - 公网 URL 输入 + 工具交叉验证",
			UserMessage(
				content=[
					TextPart(
						text=(
							"请分析这张公网图片，并和上一轮结论做冲突检查。"
							"必须先调用 get_reference_images(scene='web-contrast') 再回答。"
						)
					),
					*public_round_parts,
					TextPart(text=f"公网图像注入状态: {public_status}"),
				]
			),
		)
	)

	# Round 3: 用户再输入另一张本地图，增加多约束输出。
	round_messages.append(
		(
			"Round 3 - 第二张本地图 + 证据链输出",
			UserMessage(
				content=[
					TextPart(
						text=(
							"这是一张不同主题的本地图。请先调用 get_reference_images(scene='web-system-link')，"
							"然后输出：1) 主体对象识别 2) 与前两轮的关联点 3) 不确定项与补充证据建议。"
						)
					),
					_load_local_image_part("Web.png", detail="high"),
				]
			),
		)
	)

	# Round 4: 纯文本追问，测试模型是否仍可通过工具拉取图像证据。
	round_messages.append(
		(
			"Round 4 - 纯文本追问 + 强制工具拉取图像",
			UserMessage(
				content=(
					"请做最终审计总结：必须先调用 get_reference_images(scene='final-audit')，"
					"再输出一个包含 4 列的 Markdown 表格：图源、关键要素、风险点、可信度。"
				)
			),
		)
	)

	return round_messages


async def main() -> None:
	load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

	api_key = os.environ.get("DOUBAO_SEED_API_KEY", "")
	base_url = os.environ.get("DOUBAO_SEED_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
	model_name = os.environ.get("DOUBAO_SEED_MODEL", "")

	if not api_key:
		print("请先在 examples/.env 中配置 DOUBAO_SEED_API_KEY")
		return

	if not model_name:
		print("请先在 examples/.env 中配置 DOUBAO_SEED_MODEL")
		return

	llm_config = LLMConfig(
		model_name=model_name,
		api_key=api_key,
		base_url=base_url,
		provider="doubao_seed",
		temperature=0.3,
		top_p=0.95,
		max_tokens=4096,
		tool_choice="auto",
	)

	context = Context(
		system_prompt=(
			"你是一个严谨的多模态图像审计助手。\n"
			"规则：\n"
			"1. 每轮都要先调用 get_reference_images，再结合用户图与工具返回图回答。\n"
			"2. 回答必须区分：确定结论、推断结论、不确定项。\n"
			"3. 多轮中要引用前序轮次信息，进行冲突检查与证据链整合。\n"
			"4. 输出中文，结构化表达。"
		),
		tool_manager=[get_reference_images],
	)

	agent = Agent(
		llm_config=llm_config,
		context=context,
	)

	print("开始执行豆包图像识别多轮压力测试...\n")
	print(f"本地图片目录: {STATIC_IMG_DIR}")
	print(f"公网测试图片: {PUBLIC_IMAGE_URL}")

	round_messages = _build_round_messages()
	for round_name, user_message in round_messages:
		await _run_one_round(agent, user_message, round_name)

	_print_header("测试完成")
	print("已完成：用户输入图片 + 工具返回图片 的多轮混合验证。")


if __name__ == "__main__":
	asyncio.run(main())
