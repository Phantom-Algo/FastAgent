from typing import Any, Dict
from .openai_adapter import OpenAIAdapter


class DeepSeekAdapter(OpenAIAdapter):
    """DeepSeek OpenAI风格适配器（兼容thinking模式）。"""

    def _convert_assistant_message(self, message: Any) -> Dict[str, Any]:
        payload = super()._convert_assistant_message(message)

        # DeepSeek thinking 模式下，在工具回合期间可携带 reasoning_content 参与后续推理
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content:
            payload["reasoning_content"] = reasoning_content

        return payload
