"""
DeepSeek API 适配器实现

基于 OpenAIAdapter 扩展，针对 DeepSeek 特有行为进行适配：
1. 移除 parallel_tool_calls 参数（DeepSeek API 不支持）
2. 正确处理 reasoning_content 与 tool_calls 交织的场景
3. 历史消息中保留 reasoning_content 字段（DeepSeek 需要）
"""

import json
from typing import Any, Dict, List, Optional

from .openai_adapter import OpenAIAdapter
from ....types.context.base_context import BaseContext
from ....types.llm.base_llm_config import BaseLLMConfig
from ....types.messages.domain import AssistantMessage, UserMessage, ToolResultMessage


class DeepSeekAdapter(OpenAIAdapter):
    """
    DeepSeek API 适配器

    DeepSeek 兼容 OpenAI 格式，但有以下差异：
    - 不支持 parallel_tool_calls 参数
    - 工具调用时也会输出 reasoning_content（思维链融入工具调用）
    - 历史消息需包含 reasoning_content 以维持上下文连贯性
    """

    def _build_chat_completion_payload(
        self,
        llm_config: BaseLLMConfig,
        context: BaseContext,
        *,
        stream: bool,
    ) -> Dict[str, Any]:
        payload = super()._build_chat_completion_payload(
            llm_config, context, stream=stream
        )

        # DeepSeek 不支持 parallel_tool_calls
        payload.pop("parallel_tool_calls", None)

        return payload

    def _convert_assistant_message(self, message: AssistantMessage) -> Dict[str, Any]:
        """
        转换 AssistantMessage 为 DeepSeek API 格式。

        DeepSeek 要求历史消息中包含 reasoning_content，以维持思维链上下文连贯性。
        """
        payload = super()._convert_assistant_message(message)

        # DeepSeek 历史消息需包含 reasoning_content
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content:
            payload["reasoning_content"] = reasoning_content

        return payload
