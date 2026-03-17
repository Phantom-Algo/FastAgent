from abc import ABC, abstractmethod
from .base_adapter import IAdapter
from typing import Type

class BaseAdapterFactory(ABC):

    @classmethod
    @abstractmethod
    def register_adapter_cls(cls, provider: str, adapter_cls: Type[IAdapter]) -> bool:
        """注册适配器类"""
        ...


    @classmethod
    @abstractmethod
    def get_adapter_cls(cls, provider: str) -> Type[IAdapter]:
        """获取适配器类"""
        ...