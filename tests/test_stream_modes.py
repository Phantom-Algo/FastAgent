import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


from fast_agent.resources.adapter.already_adapter.deepseek_adapter import DeepSeekAdapter
from fast_agent.resources.adapter.already_adapter.doubao_seed_adapter import DoubaoSeedAdapter
from fast_agent.resources.adapter.already_adapter.openai_adapter import OpenAIAdapter
from fast_agent.resources.agent.agent import Agent
from fast_agent.resources.adapter.adapter_factory import AdapterFactory
from fast_agent.resources.context.context import Context
from fast_agent.resources.llm.llm_config import LLMConfig
from fast_agent.types.messages.domain import AssistantMessage, AssistantMessageChunk, ToolCall, UserMessage


class _ChunkAdapter:
    async def stream(self, llm_config, context):
        yield AssistantMessageChunk(content_delta="Hel")
        yield AssistantMessageChunk(content_delta="lo")
        yield AssistantMessage(content="Hello", finish_reason="stop", model="fake-model")

    async def invoke(self, llm_config, context):
        raise AssertionError("chunk mode should not call invoke()")


class _MessageAdapter:
    async def stream(self, llm_config, context):
        raise AssertionError("message mode should not call stream()")

    async def invoke(self, llm_config, context):
        return AssistantMessage(content="Hello", finish_reason="stop", model="fake-model")


class AgentStreamModeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.llm_config = LLMConfig(
            model_name="test-model",
            api_key="test-key",
            base_url="https://example.com/v1",
            provider="openai",
        )
        self.context = Context(system_prompt="test")
        self.user_input = UserMessage(content="hi")

    async def test_chunk_mode_emits_chunk_and_final_message_events(self):
        agent = Agent(llm_config=self.llm_config, context=self.context)

        with patch.object(AdapterFactory, "get_adapter_cls", return_value=_ChunkAdapter):
            events = [event async for event in agent.stream(self.user_input, stream_mode="chunk")]

        self.assertEqual([event.type for event in events], [
            "chunk_output_event",
            "chunk_output_event",
            "assistant_message_output_event",
            "round_stop_event",
        ])
        self.assertEqual(events[0].data.content, "Hel")
        self.assertEqual(events[1].data.content, "lo")
        self.assertEqual(events[2].data.content, "Hello")
        self.assertEqual(events[3].data.finish_reason, "stop")

    async def test_message_mode_emits_final_message_event(self):
        agent = Agent(llm_config=self.llm_config, context=self.context)

        with patch.object(AdapterFactory, "get_adapter_cls", return_value=_MessageAdapter):
            events = [event async for event in agent.stream(self.user_input, stream_mode="message")]

        self.assertEqual([event.type for event in events], [
            "assistant_message_output_event",
            "round_stop_event",
        ])
        self.assertEqual(events[0].data.content, "Hello")
        self.assertEqual(events[1].data.finish_reason, "stop")


class OpenAIAdapterInvokeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.llm_config = LLMConfig(
            model_name="test-model",
            api_key="test-key",
            base_url="https://example.com/v1",
            provider="openai",
        )
        self.context = Context(system_prompt="test")

    async def test_invoke_normalizes_non_stream_completion(self):
        completion = SimpleNamespace(
            model="response-model",
            usage=SimpleNamespace(total_tokens=42),
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    message=SimpleNamespace(
                        reasoning_content="think",
                        content=None,
                        refusal=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                function=SimpleNamespace(name="lookup", arguments='{"x": 1}'),
                            )
                        ],
                    ),
                )
            ],
        )

        class _FakeCompletions:
            def __init__(self, response):
                self.response = response
                self.last_kwargs = None

            async def create(self, **kwargs):
                self.last_kwargs = kwargs
                return self.response

        fake_completions = _FakeCompletions(completion)
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=fake_completions))

        for adapter_cls in (OpenAIAdapter, DeepSeekAdapter, DoubaoSeedAdapter):
            adapter = adapter_cls()
            with patch.object(adapter, "_build_client", return_value=fake_client):
                message = await adapter.invoke(self.llm_config, self.context)

            self.assertEqual(fake_completions.last_kwargs["stream"], False)
            self.assertEqual(message.reasoning_content, "think")
            self.assertEqual(message.finish_reason, "tool_calls")
            self.assertEqual(message.token_usage, 42)
            self.assertEqual(message.model, "response-model")
            self.assertEqual(len(message.tool_calls), 1)
            self.assertEqual(message.tool_calls[0].tool_call_id, "call_1")
            self.assertEqual(message.tool_calls[0].function_name, "lookup")
            self.assertEqual(message.tool_calls[0].function_args, {"x": 1})