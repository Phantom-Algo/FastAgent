from .base_adapter import IAdapter
from fast_agent.llm import LLMConfig, Context, AssistantMessageChunk, ToolCall, AssistantMessage
from typing import AsyncGenerator, Union, Any, Dict, List
from openai import AsyncOpenAI
import json

class OpenAIAdapter(IAdapter):
    """
    OpenAIAdapter OpenAI API 风格适配器实现
    """

    async def stream(self, llm_config: LLMConfig, context: Context) -> AsyncGenerator[Union[AssistantMessageChunk, ToolCall, AssistantMessage], None]:
        client = self._build_client(llm_config)
        request_payload = self._build_chat_completion_payload(llm_config, context, stream=True)

        stream = await client.chat.completions.create(**request_payload)

        reasoning_content = ""
        content = ""
        refusal = ""
        finish_reason = "unknown"
        response_model = None

        tool_call_chunks: Dict[int, Dict[str, Any]] = {}

        async for chunk in stream:
            response_model = getattr(chunk, "model", response_model)

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue

            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason

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
                if function is not None:
                    function_name = getattr(function, "name", None)
                    if function_name:
                        item["name"] = function_name

                    function_arguments = getattr(function, "arguments", None)
                    if function_arguments:
                        item["arguments"] += function_arguments

        tool_calls = self._build_tool_calls_from_stream_buffer(tool_call_chunks)
        for tool_call in tool_calls:
            yield tool_call

        final_message = AssistantMessage(
            reasoning_content=reasoning_content or None,
            content=content or None,
            refusal=refusal or None,
            tool_calls=tool_calls or None,
            finish_reason=finish_reason,
            model=response_model,
        )

        yield final_message
    
    async def invoke(self, llm_config: LLMConfig, context: Context) -> AssistantMessage:
        client = self._build_client(llm_config)
        request_payload = self._build_chat_completion_payload(llm_config, context, stream=False)

        response = await client.chat.completions.create(**request_payload)

        choice = response.choices[0]
        message = choice.message

        tool_calls = []
        for tool_call in (getattr(message, "tool_calls", None) or []):
            tool_calls.append(
                ToolCall(
                    tool_call_id=tool_call.id,
                    function_name=tool_call.function.name,
                    function_args=self._safe_json_loads(tool_call.function.arguments),
                )
            )

        usage = getattr(response, "usage", None)
        token_usage = getattr(usage, "total_tokens", None) if usage else None

        return AssistantMessage(
            reasoning_content=getattr(message, "reasoning_content", None),
            content=getattr(message, "content", None),
            refusal=getattr(message, "refusal", None),
            tool_calls=tool_calls or None,
            finish_reason=choice.finish_reason or "unknown",
            token_usage=token_usage,
            model=getattr(response, "model", None),
        )

    def _build_client(self, llm_config: LLMConfig) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
        )

    def _build_chat_completion_payload(self, llm_config: LLMConfig, context: Context, *, stream: bool) -> Dict[str, Any]:
        messages = self._convert_messages(context)

        payload: Dict[str, Any] = {
            "model": llm_config.model_name,
            "messages": messages,
            "temperature": llm_config.temperature,
            "top_p": llm_config.top_p,
            "max_tokens": llm_config.max_tokens,
            "frequency_penalty": llm_config.frequency_penalty,
            "presence_penalty": llm_config.presence_penalty,
            "stream": stream,
        }

        if llm_config.stop_sequences:
            payload["stop"] = llm_config.stop_sequences

        tools = context.tools.get_tools()
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

    def _convert_messages(self, context: Context) -> List[Dict[str, Any]]:
        openai_messages: List[Dict[str, Any]] = []

        system_prompt = context.system_prompt.get_system_prompt()
        if system_prompt:
            openai_messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        for message in context.work_messages.get_messages():
            role = getattr(message, "role", None)

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

    def _convert_user_message(self, message: Any) -> Dict[str, Any]:
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

            if part_type == "image":
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

    def _convert_assistant_message(self, message: Any) -> Dict[str, Any]:
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

    def _convert_tool_result_message(self, message: Any) -> Dict[str, Any]:
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

        raw_args = raw_args.strip()
        if not raw_args:
            return {}

        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        except json.JSONDecodeError:
            return {"_raw": raw_args}

    def _build_tool_calls_from_stream_buffer(self, tool_call_chunks: Dict[int, Dict[str, Any]]) -> List[ToolCall]:
        tool_calls: List[ToolCall] = []
        for index in sorted(tool_call_chunks.keys()):
            item = tool_call_chunks[index]
            function_name = item.get("name")
            if not function_name:
                continue

            tool_call_payload = {
                "function_name": function_name,
                "function_args": self._safe_json_loads(item.get("arguments")),
            }
            tool_call_id = item.get("id")
            if tool_call_id:
                tool_call_payload["tool_call_id"] = tool_call_id

            tool_calls.append(ToolCall(**tool_call_payload))

        return tool_calls