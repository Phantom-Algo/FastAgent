from typing import Type, Dict
from .base_adapter import IAdapter
from .openai_adapter import OpenAIAdapter
from .deepseek_adapter import DeepSeekAdapter
from enum import Enum

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"

class AdapterFactory:
    """适配器工厂类"""
    _mapping: Dict[str, Type[IAdapter]] = {
        LLMProvider.OPENAI.value: OpenAIAdapter,
        LLMProvider.DEEPSEEK.value: DeepSeekAdapter,
    }

    @classmethod
    def register_adapter_cls(cls, provider: str, adapter_cls: Type[IAdapter]):
        """注册适配器类"""
        cls._mapping[provider] = adapter_cls

    @classmethod
    def get_adapter_cls(cls, provider: str) -> Type[IAdapter]:
        """获取适配器类"""
        adapter_cls = cls._mapping.get(provider)
        if adapter_cls is None:
            raise ValueError(f"No adapter class registered for provider: {provider}")
        return adapter_cls
    
    @classmethod
    def get_openai_adapter_cls(cls) -> Type[IAdapter]:
        """获取 OpenAI 适配器类"""
        return cls.get_adapter_cls(provider=LLMProvider.OPENAI.value)
    

    
        
    
        
    
    