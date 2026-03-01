from typing import Optional, List
from .messages_model import BaseMessage

class Messages:
    """
    Messages 消息类，用于定义与管理消息
    """
    def __init__(
        self,
        messages: Optional[List[BaseMessage]] = None
    ):
        if messages is None:
            self.messages = []
        else:
            self.messages = messages

    # === 增 ===
    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)

    # === 删 ===
    def remove_message_by_id(self, id: str) -> Optional[BaseMessage]:
        for index, message in enumerate(self.messages):
            if message.id == id:
                return self.messages.pop(index)
        return None
    
    def clear_messages(self) -> None:
        self.messages.clear()

    def pop_message(self) -> Optional[BaseMessage]:
        if self.messages:
            return self.messages.pop()
        return None

    # === 查 ===
    def get_messages(self) -> List[BaseMessage]:
        return self.messages
    
    def get_message_count(self) -> int:
        return len(self.messages)
    
    def get_message_by_id(self, id: str) -> Optional[BaseMessage]:
        for message in self.messages:
            if message.id == id:
                return message
        return None
    
    def get_last_message(self) -> Optional[BaseMessage]:
        if self.messages:
            return self.messages[-1]
        return None
    
    # === 改 ===
    def update_message_by_id(self, id: str, new_message: BaseMessage) -> bool:
        for index, message in enumerate(self.messages):
            if message.id == id:
                self.messages[index] = new_message
                return True
        return False
    
    def update_messages(self, new_messages: List[BaseMessage]) -> None:
        self.messages = new_messages