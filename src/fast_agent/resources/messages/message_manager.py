from ...types.messages.base_message_manager import BaseMessageManager
from ...types.messages.base_message import BaseMessage
from ...types.messages.domain.assistant_message import AssistantMessage
from ...types.messages.domain.message_round import MessageRound
from ...types.messages.domain.tool_result_message import ToolResultMessage
from ...types.messages.domain.user_message import UserMessage
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

    def get_round_count(self) -> int:
        return len(self._build_rounds())

    def get_rounds(
        self,
        start_round_index: Optional[int] = None,
        end_round_index: Optional[int] = None,
    ) -> List[MessageRound]:
        rounds = self._build_rounds()
        if not rounds:
            return []

        start = 0 if start_round_index is None else start_round_index
        end = len(rounds) - 1 if end_round_index is None else end_round_index
        self._validate_round_range(start, end, len(rounds))
        return rounds[start:end + 1]


    # === 改 ===
    def update_message_by_id(self, id: str, new_message: BaseMessage) -> bool:
        for index, message in enumerate(self.messages):
            if message.id == id:
                self.messages[index] = new_message.model_copy(update={"id": id})
                return True
        return False

    def update_messages(self, new_messages: List[BaseMessage]) -> None:
        self.messages = list(new_messages)

    def remove_rounds(self, start_round_index: int, end_round_index: int) -> List[MessageRound]:
        rounds = self._build_rounds()
        if not rounds:
            return []

        self._validate_round_range(start_round_index, end_round_index, len(rounds))
        removed_rounds = rounds[start_round_index:end_round_index + 1]
        start_message_index = removed_rounds[0].start_message_index
        end_message_index = removed_rounds[-1].end_message_index
        del self.messages[start_message_index:end_message_index + 1]
        return removed_rounds

    def _build_rounds(self) -> List[MessageRound]:
        rounds: List[MessageRound] = []
        cursor = 0
        round_index = 0
        total_messages = len(self.messages)

        while cursor < total_messages:
            message = self.messages[cursor]
            if not isinstance(message, UserMessage):
                cursor += 1
                continue

            start_message_index = cursor
            user_message = message
            cursor += 1

            if cursor >= total_messages:
                break

            assistant_messages: List[AssistantMessage] = []
            tool_result_messages: List[ToolResultMessage] = []

            while True:
                assistant_message = self._get_message_as(cursor, AssistantMessage, "AssistantMessage")
                if assistant_message is None:
                    return rounds

                assistant_messages.append(assistant_message)
                cursor += 1

                if not assistant_message.tool_calls:
                    end_message_index = cursor - 1
                    rounds.append(
                        MessageRound(
                            round_index=round_index,
                            start_message_index=start_message_index,
                            end_message_index=end_message_index,
                            messages=list(self.messages[start_message_index:end_message_index + 1]),
                            user_message=user_message,
                            assistant_messages=list(assistant_messages),
                            tool_result_messages=list(tool_result_messages),
                        )
                    )
                    round_index += 1
                    break

                tool_result_count = 0
                while cursor < total_messages and isinstance(self.messages[cursor], ToolResultMessage):
                    tool_result_messages.append(self.messages[cursor])
                    tool_result_count += 1
                    cursor += 1

                if tool_result_count == 0:
                    if cursor >= total_messages:
                        return rounds
                    raise ValueError(
                        f"Invalid round sequence at message index {cursor}: expected ToolResultMessage after AssistantMessage tool call."
                    )

                if cursor >= total_messages:
                    return rounds

        return rounds

    def _get_message_as(self, index: int, expected_type: type, expected_name: str):
        if index >= len(self.messages):
            return None

        message = self.messages[index]
        if not isinstance(message, expected_type):
            raise ValueError(
                f"Invalid round sequence at message index {index}: expected {expected_name}, got {type(message).__name__}."
            )
        return message

    def _validate_round_range(self, start_round_index: int, end_round_index: int, round_count: int) -> None:
        if start_round_index < 0 or end_round_index < 0:
            raise IndexError("Round index must be greater than or equal to 0.")
        if start_round_index > end_round_index:
            raise IndexError("start_round_index must be less than or equal to end_round_index.")
        if end_round_index >= round_count:
            raise IndexError(
                f"Round index out of range. Current round count is {round_count}, received {start_round_index}-{end_round_index}."
            )

    