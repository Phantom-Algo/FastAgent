import json
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Union

from openai import AsyncOpenAI

from ....types.adapter.base_adapter import IAdapter
from ....types.context.base_context import BaseContext
from ....types.embeddings.base_embedding_config import BaseEmbeddingConfig
from ....types.embeddings.domain import EmbeddingResponse, EmbeddingUsage, EmbeddingVector
from ....types.llm.base_llm_config import BaseLLMConfig
from ....types.messages.base_message_manager import BaseMessageManager
from ....types.messages.domain import (
    AssistantMessage, 
    AssistantMessageChunk, 
    ToolCall, 
    ToolResultMessage, 
    UserMessage,
    BasePart,
    TextPart,
    ImagePart
)
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

        yield self._build_assistant_message(
            reasoning_content=reasoning_content,
            content=content,
            refusal=refusal,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            model=response_model,
        )

    async def invoke(self, llm_config: BaseLLMConfig, context: BaseContext) -> AssistantMessage:
        client = self._build_client(llm_config)
        request_payload = self._build_chat_completion_payload(llm_config, context, stream=False)

        completion = await client.chat.completions.create(**request_payload)
        return self._build_assistant_message_from_completion(completion)

    async def embed(self, embedding_config: BaseEmbeddingConfig, inputs: Union[str, List[str]]) -> EmbeddingResponse:
        client = self._build_client(embedding_config)
        request_payload = self._build_embedding_payload(
            embedding_config,
            self._normalize_embedding_inputs(inputs),
        )

        response = await client.embeddings.create(**request_payload)
        return self._build_embedding_response(response)

    def _build_client(self, config: Union[BaseLLMConfig, BaseEmbeddingConfig]) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    def _build_embedding_payload(
        self,
        embedding_config: BaseEmbeddingConfig,
        inputs: Union[str, List[str]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": embedding_config.model_name,
            "input": inputs,
            "encoding_format": embedding_config.encoding_format,
        }

        if embedding_config.dimensions is not None:
            payload["dimensions"] = embedding_config.dimensions

        if embedding_config.user:
            payload["user"] = embedding_config.user

        if embedding_config.extra_body:
            payload["extra_body"] = embedding_config.extra_body

        return payload

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

        if llm_config.response_format:
            payload["response_format"] = llm_config.response_format

        if llm_config.extra_body:
            payload["extra_body"] = llm_config.extra_body

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

        openai_parts = self._convert_content_parts(content)

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
        if isinstance(message.content, str):
            content: Union[str, List[Dict[str, Any]]] = message.content
        elif isinstance(message.content, list):
            content = self._convert_content_parts(message.content)
            if not content:
                content = ""
        else:
            content = json.dumps(message.content, ensure_ascii=False, default=str)

        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "name": message.name,
            "content": content,
        }

    def _convert_content_parts(self, content_parts: List[BasePart]) -> List[Dict[str, Any]]:
        if not isinstance(content_parts, list):
            return []

        openai_parts: List[Dict[str, Any]] = []
        for part in content_parts:

            if isinstance(part, TextPart):
                text = part.text
                if text:
                    openai_parts.append(
                        {
                            "type": "text",
                            "text": text,
                        }
                    )
                continue

            if not isinstance(part, ImagePart):
                continue

            image_url = None

            # 优先公网 URL
            if part.url:
                image_url = part.url

            # Base64 编码的图片数据
            elif part.base64_data and part.mime_type:
                image_url = f"data:{part.mime_type};base64,{part.base64_data}"

            # 若均无则跳过
            if image_url is None:
                continue

            image_part: Dict[str, Any] = {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            }

            # 清晰度
            if part.detail:
                image_part["image_url"]["detail"] = part.detail

            openai_parts.append(image_part)

        return openai_parts

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

    def _normalize_embedding_inputs(self, inputs: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(inputs, str):
            return inputs

        if not isinstance(inputs, list):
            raise TypeError("Embedding inputs must be a string or a list of strings.")

        if not inputs:
            raise ValueError("Embedding inputs must not be empty.")

        if any(not isinstance(item, str) for item in inputs):
            raise TypeError("Embedding input list must contain only strings.")

        return inputs

    def _build_embedding_response(self, response: Any) -> EmbeddingResponse:
        data = [
            EmbeddingVector(
                object=getattr(item, "object", "embedding"),
                index=getattr(item, "index", index),
                embedding=getattr(item, "embedding"),
            )
            for index, item in enumerate(getattr(response, "data", None) or [])
            if getattr(item, "embedding", None) is not None
        ]

        usage_payload = getattr(response, "usage", None)
        usage = None
        if usage_payload is not None:
            usage = EmbeddingUsage(
                prompt_tokens=getattr(usage_payload, "prompt_tokens", 0) or 0,
                total_tokens=getattr(usage_payload, "total_tokens", 0) or 0,
            )

        return EmbeddingResponse(
            object=getattr(response, "object", "list"),
            data=data,
            model=getattr(response, "model", None),
            usage=usage,
        )

    def _build_assistant_message_from_completion(self, completion: Any) -> AssistantMessage:
        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise ValueError("OpenAI chat completion returned no choices.")

        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            raise ValueError("OpenAI chat completion returned no message payload.")

        tool_calls: List[ToolCall] = []
        for tool_call in getattr(message, "tool_calls", None) or []:
            function = getattr(tool_call, "function", None)
            if function is None:
                continue

            function_name = getattr(function, "name", None)
            if not function_name:
                continue

            tool_calls.append(
                ToolCall(
                    tool_call_id=getattr(tool_call, "id", None),
                    function_name=function_name,
                    function_args=self._safe_json_loads(getattr(function, "arguments", None)),
                )
            )

        usage = getattr(completion, "usage", None)
        token_usage = getattr(usage, "total_tokens", None)

        return self._build_assistant_message(
            reasoning_content=getattr(message, "reasoning_content", None),
            content=getattr(message, "content", None),
            refusal=getattr(message, "refusal", None),
            tool_calls=tool_calls,
            finish_reason=self._normalize_finish_reason(getattr(choice, "finish_reason", None)),
            token_usage=token_usage,
            model=getattr(completion, "model", None),
        )

    def _build_assistant_message(
        self,
        *,
        reasoning_content: Optional[str] = None,
        content: Optional[str] = None,
        refusal: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None,
        finish_reason: Literal["unknown", "stop", "length", "tool_calls", "content_filter", "balance", "error"] = "unknown",
        token_usage: Optional[int] = None,
        model: Optional[str] = None,
    ) -> AssistantMessage:
        message_payload: Dict[str, Any] = {
            "finish_reason": finish_reason,
            "token_usage": token_usage,
            "model": model,
        }

        if reasoning_content:
            message_payload["reasoning_content"] = reasoning_content
        if content:
            message_payload["content"] = content
        if refusal:
            message_payload["refusal"] = refusal
        if tool_calls:
            message_payload["tool_calls"] = tool_calls

        has_message_body = any(
            [
                message_payload.get("reasoning_content"),
                message_payload.get("content"),
                message_payload.get("refusal"),
                message_payload.get("tool_calls"),
            ]
        )
        if not has_message_body:
            raise ValueError("Assistant message is empty after OpenAI response normalization.")

        return AssistantMessage(**message_payload)

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