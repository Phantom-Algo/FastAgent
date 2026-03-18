"""
Doubao Seed API 适配器实现

基于 OpenAIAdapter 扩展，针对豆包（Seed）在推理与工具调用混合场景的行为进行适配：
1. 工具调用轮次中保留 reasoning_content 上下文
2. 移除部分 OpenAI 扩展参数，提升兼容性
"""

from typing import Any, Dict

from .openai_adapter import OpenAIAdapter
from ....types.messages.domain import AssistantMessage


class DoubaoSeedAdapter(OpenAIAdapter):
	"""
	豆包 Seed 适配器。

	豆包与 OpenAI Chat Completions 高度兼容，但在多轮工具调用时，
	将 reasoning_content 持续纳入历史上下文可以显著提升后续推理连贯性。
	"""

	def _convert_assistant_message(self, message: AssistantMessage) -> Dict[str, Any]:
		payload = super()._convert_assistant_message(message)

		# 保留历史推理内容，支持思维链与工具调用混合的后续轮次。
		if message.reasoning_content:
			payload["reasoning_content"] = message.reasoning_content

		return payload
