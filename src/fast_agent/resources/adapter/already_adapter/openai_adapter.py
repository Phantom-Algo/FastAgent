import json
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Union

from openai import AsyncOpenAI

from ....types.adapter.base_adapter import IAdapter
from ....types.context.base_context import BaseContext
from ....types.llm.base_llm_config import BaseLLMConfig
from ....types.messages.base_message import BaseMessage
from ....types.messages.base_message_manager import BaseMessageManager
from ....types.messages.domain import AssistantMessage, AssistantMessageChunk, ToolCall, ToolResultMessage, UserMessage
from ....types.system_prompt.base_system_prompt import BaseSystemPrompt
from ....types.tool.base_tool import BaseTool
from ....types.tool.base_tool_manager import BaseToolManager

class OpenAIAdapter(IAdapter):
    """
    OpenAI API 风格适配器实现
    """

    async def stream(self, llm_config: BaseLLMConfig, context: BaseContext) -> AsyncGenerator[Union[AssistantMessageChunk, ToolCall, AssistantMessage], None]:
        client = self._build_client(llm_config)
        request_payload = self._build_chat_completion_payload(llm_config, context, stream=True)

        stream = await client.chat.completions.create(**request_payload)

        reasoning_content = ""
        content = ""
        refusal = ""
        finish_reason: Literal["unknown", "stop", "length", "tool_calls", "content_filter", "balance", "error"] = "unknown"
        response_model = None
        tool_call_chunks: Dict[int, Dict[str, Any]] = {}

        async for chunk in stream:
            response_model = getattr(chunk, "model", response_model)

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue

            choice = choices[0]
            delta = getattr(choice, "delta", None)

            normalized_finish_reason = self._normalize_finish_reason(getattr(choice, "finish_reason", None))
            if normalized_finish_reason != "unknown":
                finish_reason = normalized_finish_reason

            if delta is None:
                continue

            reasoning_delta = getattr(delta, "reasoning_content", None)
            if reasoning_delta:
                reasoning_content += reasoning_delta
                yield AssistantMessageChunk(reasoning_content_delta=reasoning_delta)

            content_delta = getattr(delta, "content", None)
            if content_delta:
                content += content_delta
                yield AssistantMessageChunk(content_delta=content_delta)

            refusal_delta = getattr(delta, "refusal", None)
            if refusal_delta:
                refusal += refusal_delta
                yield AssistantMessageChunk(refusal_delta=refusal_delta)

            delta_tool_calls = getattr(delta, "tool_calls", None) or []
            for delta_tool_call in delta_tool_calls:
                index = getattr(delta_tool_call, "index", 0)
                item = tool_call_chunks.setdefault(
                    index,
                    {
                        "id": None,
                        "name": None,
                        "arguments": "",
                    },
                )

                tool_call_id = getattr(delta_tool_call, "id", None)
                if tool_call_id:
                    item["id"] = tool_call_id

                function = getattr(delta_tool_call, "function", None)
                if function is None:
                    continue

                function_name = getattr(function, "name", None)
                if function_name:
                    item["name"] = function_name

                function_arguments = getattr(function, "arguments", None)
                if function_arguments:
                    item["arguments"] += function_arguments

        tool_calls = self._build_tool_calls_from_stream_buffer(tool_call_chunks)
        for tool_call in tool_calls:
            yield tool_call

        final_message_payload: Dict[str, Any] = {
            "finish_reason": finish_reason,
            "model": response_model,
        }
        if reasoning_content:
            final_message_payload["reasoning_content"] = reasoning_content
        if content:
            final_message_payload["content"] = content
        if refusal:
            final_message_payload["refusal"] = refusal
        if tool_calls:
            final_message_payload["tool_calls"] = tool_calls

        if len(final_message_payload) == 2:
            final_message_payload["refusal"] = ""

        yield AssistantMessage(**final_message_payload)

    def _build_client(self, llm_config: BaseLLMConfig) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
        )

    def _build_chat_completion_payload(
        self,
        llm_config: BaseLLMConfig,
        context: BaseContext,
        *,
        stream: bool,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": llm_config.model_name,
            "messages": self._convert_messages(context),
            "temperature": llm_config.temperature,
            "top_p": llm_config.top_p,
            "max_tokens": llm_config.max_tokens,
            "frequency_penalty": llm_config.frequency_penalty,
            "presence_penalty": llm_config.presence_penalty,
            "stream": stream,
        }

        if llm_config.stop_sequences:
            payload["stop"] = llm_config.stop_sequences

        tools = self._get_context_tools(context)
        if tools:
            payload["tools"] = [tool.to_openai_schema() for tool in tools]
            payload["parallel_tool_calls"] = llm_config.parallel_tool_calls
            payload["tool_choice"] = self._convert_tool_choice(llm_config.tool_choice)

        return payload

    def _convert_tool_choice(self, tool_choice: str) -> Any:
        if tool_choice in ("auto", "none", "required"):
            return tool_choice

        return {
            "type": "function",
            "function": {
                "name": tool_choice,
            },
        }

    def _convert_messages(self, context: BaseContext) -> List[Dict[str, Any]]:
        openai_messages: List[Dict[str, Any]] = []

        system_prompt = self._get_system_prompt(context)
        if system_prompt:
            openai_messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        for message in self._get_context_messages(context):
            role = message.role

            if role == "user":
                openai_messages.append(self._convert_user_message(message))
                continue

            if role == "assistant":
                openai_messages.append(self._convert_assistant_message(message))
                continue

            if role == "tool_result":
                openai_messages.append(self._convert_tool_result_message(message))
                continue

            raise ValueError(f"Unsupported message role for OpenAI chat completions: {role}")

        return openai_messages

    def _convert_user_message(self, message: UserMessage) -> Dict[str, Any]:
        content = getattr(message, "content", "")

        if isinstance(content, str):
            return {
                "role": "user",
                "content": content,
            }

        openai_parts: List[Dict[str, Any]] = []
        for part in content:
            part_type = getattr(part, "type", None)

            if part_type == "text":
                openai_parts.append(
                    {
                        "type": "text",
                        "text": part.text,
                    }
                )
                continue

            if part_type != "image":
                continue

            image_url = None
            if getattr(part, "url", None):
                image_url = part.url
            elif getattr(part, "file_url", None):
                image_url = part.file_url
            elif getattr(part, "base64_data", None) and getattr(part, "mime_type", None):
                image_url = f"data:{part.mime_type};base64,{part.base64_data}"

            if image_url is None:
                continue

            image_part: Dict[str, Any] = {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            }
            if getattr(part, "detail", None):
                image_part["image_url"]["detail"] = part.detail

            openai_parts.append(image_part)

        return {
            "role": "user",
            "content": openai_parts,
        }

    def _convert_assistant_message(self, message: AssistantMessage) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "role": "assistant",
            "content": getattr(message, "content", None),
        }

        if getattr(message, "tool_calls", None):
            payload["tool_calls"] = [
                {
                    "id": tool_call.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function_name,
                        "arguments": json.dumps(tool_call.function_args, ensure_ascii=False),
                    },
                }
                for tool_call in message.tool_calls
            ]

        if payload.get("content") is None and payload.get("tool_calls") is None:
            payload["content"] = ""

        return payload

    def _convert_tool_result_message(self, message: ToolResultMessage) -> Dict[str, Any]:
        content = message.content if isinstance(message.content, str) else json.dumps(message.content, ensure_ascii=False)
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "name": message.name,
            "content": content,
        }

    def _safe_json_loads(self, raw_args: Any) -> Dict[str, Any]:
        if raw_args is None:
            return {}

        if isinstance(raw_args, dict):
            return raw_args

        if not isinstance(raw_args, str):
            return {}

        normalized_raw_args = raw_args.strip()
        if not normalized_raw_args:
            return {}

        try:
            parsed = json.loads(normalized_raw_args)
        except json.JSONDecodeError:
            return {"_raw": normalized_raw_args}

        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}

    def _build_tool_calls_from_stream_buffer(self, tool_call_chunks: Dict[int, Dict[str, Any]]) -> List[ToolCall]:
        tool_calls: List[ToolCall] = []
        for index in sorted(tool_call_chunks.keys()):
            item = tool_call_chunks[index]
            function_name = item.get("name")
            if not function_name:
                continue

            tool_call_payload: Dict[str, Any] = {
                "function_name": function_name,
                "function_args": self._safe_json_loads(item.get("arguments")),
            }

            tool_call_id = item.get("id")
            if tool_call_id:
                tool_call_payload["tool_call_id"] = tool_call_id

            tool_calls.append(ToolCall(**tool_call_payload))

        return tool_calls

    def _normalize_finish_reason(
        self,
        finish_reason: Any,
    ) -> Literal["unknown", "stop", "length", "tool_calls", "content_filter", "balance", "error"]:
        if finish_reason in {"stop", "length", "tool_calls", "content_filter", "balance", "error"}:
            return finish_reason
        return "unknown"

    def _get_system_prompt(self, context: BaseContext) -> str:
        system_prompt: Optional[BaseSystemPrompt] = context.get_system_prompt()
        if system_prompt is None:
            return ""
        return system_prompt.get_system_prompt() or ""

    def _get_context_messages(self, context: BaseContext) -> List[Union[UserMessage, AssistantMessage, ToolResultMessage]]:
        work_message_manager: Optional[BaseMessageManager] = context.get_work_message_manager()
        if work_message_manager is None:
            return []
        return work_message_manager.get_messages()

    def _get_context_tools(self, context: BaseContext) -> List[BaseTool]:
        tool_manager: Optional[BaseToolManager] = context.get_tool_manager()
        if tool_manager is None:
            return []
        return tool_manager.get_tools()