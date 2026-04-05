from abc import ABC, abstractmethod
from typing import List, Optional

from .base_message import BaseMessage
from .domain.message_round import MessageRound

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

    @abstractmethod
    def get_round_count(self) -> int:
        ...

    @abstractmethod
    def get_rounds(
        self,
        start_round_index: Optional[int] = None,
        end_round_index: Optional[int] = None,
    ) -> List[MessageRound]:
        ...
    


    # === 改 ===
    @abstractmethod
    def update_message_by_id(self, id: str, new_message: BaseMessage) -> bool:
        ...
    
    @abstractmethod
    def update_messages(self, new_messages: List[BaseMessage]) -> None:
        ...

    @abstractmethod
    def remove_rounds(self, start_round_index: int, end_round_index: int) -> List[MessageRound]:
        ...