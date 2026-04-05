import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


from fast_agent.resources.adapter.already_adapter.openai_adapter import OpenAIAdapter
from fast_agent.resources.embeddings.embedding_config import EmbeddingConfig


class OpenAIAdapterEmbeddingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.adapter = OpenAIAdapter()
        self.embedding_config = EmbeddingConfig(
            model_name="text-embedding-3-small",
            api_key="test-key",
            base_url="https://example.com/v1",
            provider="openai",
            dimensions=256,
            user="tester",
        )

    async def test_embed_normalizes_openai_response(self):
        response = SimpleNamespace(
            object="list",
            model="text-embedding-3-small",
            data=[
                SimpleNamespace(object="embedding", index=0, embedding=[0.1, 0.2]),
                SimpleNamespace(object="embedding", index=1, embedding=[0.3, 0.4]),
            ],
            usage=SimpleNamespace(prompt_tokens=6, total_tokens=6),
        )

        class _FakeEmbeddings:
            def __init__(self, response):
                self.response = response
                self.last_kwargs = None

            async def create(self, **kwargs):
                self.last_kwargs = kwargs
                return self.response

        fake_embeddings = _FakeEmbeddings(response)
        fake_client = SimpleNamespace(embeddings=fake_embeddings)

        with patch.object(self.adapter, "_build_client", return_value=fake_client):
            embedding_response = await self.adapter.embed(self.embedding_config, ["hello", "world"])

        self.assertEqual(fake_embeddings.last_kwargs["model"], "text-embedding-3-small")
        self.assertEqual(fake_embeddings.last_kwargs["input"], ["hello", "world"])
        self.assertEqual(fake_embeddings.last_kwargs["dimensions"], 256)
        self.assertEqual(fake_embeddings.last_kwargs["encoding_format"], "float")
        self.assertEqual(fake_embeddings.last_kwargs["user"], "tester")
        self.assertEqual(embedding_response.model, "text-embedding-3-small")
        self.assertEqual(len(embedding_response.data), 2)
        self.assertEqual(embedding_response.data[0].embedding, [0.1, 0.2])
        self.assertEqual(embedding_response.usage.prompt_tokens, 6)
        self.assertEqual(embedding_response.usage.total_tokens, 6)

    async def test_embed_accepts_single_string_input(self):
        response = SimpleNamespace(
            object="list",
            model="text-embedding-3-small",
            data=[SimpleNamespace(object="embedding", index=0, embedding=[0.1, 0.2])],
            usage=None,
        )

        class _FakeEmbeddings:
            def __init__(self, response):
                self.response = response
                self.last_kwargs = None

            async def create(self, **kwargs):
                self.last_kwargs = kwargs
                return self.response

        fake_embeddings = _FakeEmbeddings(response)
        fake_client = SimpleNamespace(embeddings=fake_embeddings)

        with patch.object(self.adapter, "_build_client", return_value=fake_client):
            embedding_response = await self.adapter.embed(self.embedding_config, "hello")

        self.assertEqual(fake_embeddings.last_kwargs["input"], "hello")
        self.assertEqual(len(embedding_response.data), 1)

    def test_embed_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            self.adapter._normalize_embedding_inputs([])

        with self.assertRaises(TypeError):
            self.adapter._normalize_embedding_inputs(["ok", 1])