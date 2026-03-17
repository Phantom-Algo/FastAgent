from abc import ABC, abstractmethod
from .base_event import BaseEvent
from typing import Optional

class BaseEventChannel(ABC):
    """
    BaseEventChannel 定义了事件通道的抽象类
    """

    @abstractmethod
    async def send_event(self, event: BaseEvent) -> None:
        ...

    @abstractmethod
    async def receive_event(self, timeout: Optional[int] = None) -> BaseEvent:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    @property
    def is_closed(self) -> bool:
        ...

    @abstractmethod
    def task_done(self) -> None:
        ...