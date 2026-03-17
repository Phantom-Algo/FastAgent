from abc import ABC, abstractmethod
from .base_message import BaseMessage
from typing import List, Optional

class BaseMessageManager(ABC):

    # === 增 ===
    @abstractmethod
    def add_message(self, message: BaseMessage) -> None:
        ...

    @abstractmethod
    def add_messages(self, new_messages: List[BaseMessage]) -> None:
        ...



    # === 删 ===
    @abstractmethod
    def remove_message_by_id(self, id: str) -> Optional[BaseMessage]:
        ...
    
    @abstractmethod
    def clear_messages(self) -> None:
        ...

    @abstractmethod
    def pop_message(self) -> Optional[BaseMessage]:
        ...


    # === 查 ===
    @abstractmethod
    def get_messages(self) -> List[BaseMessage]:
        ...
    
    @abstractmethod
    def get_message_count(self) -> int:
        ...
    
    @abstractmethod
    def get_message_by_id(self, id: str) -> Optional[BaseMessage]:
        ...
    
    @abstractmethod
    def get_last_message(self) -> Optional[BaseMessage]:
        ...
    


    # === 改 ===
    @abstractmethod
    def update_message_by_id(self, id: str, new_message: BaseMessage) -> bool:
        ...
    
    @abstractmethod
    def update_messages(self, new_messages: List[BaseMessage]) -> None:
        ...