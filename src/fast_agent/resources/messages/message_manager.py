from ...types.messages.base_message_manager import BaseMessageManager
from ...types.messages.base_message import BaseMessage
from typing import List, Optional


class MessageManager(BaseMessageManager):
    def __init__(
        self,
        messages: Optional[List[BaseMessage]] = None
    ):
        self.messages = messages if messages is not None else []

    # === 增 ===
    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)

    def add_messages(self, new_messages: List[BaseMessage]) -> None:
        self.messages.extend(new_messages)


    # === 删 ===
    def remove_message_by_id(self, id: str) -> Optional[BaseMessage]:
        for index, message in enumerate(self.messages):
            if message.id == id:
                return self.messages.pop(index)
        return None

    def clear_messages(self) -> None:
        self.messages.clear()

    def pop_message(self) -> Optional[BaseMessage]:
        if not self.messages:
            return None
        return self.messages.pop()


    # === 查 ===
    def get_messages(self) -> List[BaseMessage]:
        return list(self.messages)

    def get_message_count(self) -> int:
        return len(self.messages)

    def get_message_by_id(self, id: str) -> Optional[BaseMessage]:
        for message in self.messages:
            if message.id == id:
                return message
        return None

    def get_last_message(self) -> Optional[BaseMessage]:
        if not self.messages:
            return None
        return self.messages[-1]


    # === 改 ===
    def update_message_by_id(self, id: str, new_message: BaseMessage) -> bool:
        for index, message in enumerate(self.messages):
            if message.id == id:
                self.messages[index] = new_message.model_copy(update={"id": id})
                return True
        return False

    def update_messages(self, new_messages: List[BaseMessage]) -> None:
        self.messages = list(new_messages)

    