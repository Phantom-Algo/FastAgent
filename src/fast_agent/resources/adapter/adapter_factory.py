from ...types.adapter.base_adapter_factory import BaseAdapterFactory
from ...types.llm.enum.llm_provider_enum import LLMProviderEnum
from ...types.adapter.base_adapter import IAdapter
from .already_adapter.openai_adapter import OpenAIAdapter
from .already_adapter.deepseek_adapter import DeepSeekAdapter
from .already_adapter.doubao_seed_adapter import DoubaoSeedAdapter
from typing import Type


class AdapterFactory(BaseAdapterFactory):

    _mapping = {
        LLMProviderEnum.OPENAI.value: OpenAIAdapter,
        LLMProviderEnum.DEEPSEEK.value: DeepSeekAdapter,
        LLMProviderEnum.DOUBAO_SEED.value: DoubaoSeedAdapter,
    }

    @classmethod
    def register_adapter_cls(cls, provider, adapter_cls) -> bool:
        if provider in cls._mapping:
            return False
        cls._mapping[provider] = adapter_cls
        return True
        

    @classmethod
    def get_adapter_cls(cls, provider):
        adapter_cls: Type[IAdapter] = cls._mapping.get(provider, None)
        if adapter_cls is None:
            raise ValueError(f"No adapter found for provider: {provider}")
        return adapter_cls
    

    @classmethod
    def get_openai_adapter_cls(cls):
        return cls.get_adapter_cls(LLMProviderEnum.OPENAI.value)
