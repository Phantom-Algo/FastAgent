from abc import ABC, abstractmethod
from typing import AsyncGenerator, Literal, Dict, Optional
from ..messages.domain import UserMessage
from .event.base_event import BaseEvent
from .snapshot.base_snapshot import BaseSnapshot
from ..tool.domain.guard_policy import GuardPolicyHumanResponseSchema

class BaseAgent(ABC):

    @abstractmethod
    async def stream(
        self, 
        user_input: UserMessage, 
        stream_mode: Literal["chunk", "message"] = "chunk"
    ) -> AsyncGenerator[BaseEvent, None]:
        ...


    @abstractmethod
    async def resume_stream(
        self,
        snapshot: BaseSnapshot,
        human_response: Optional[Dict[str, GuardPolicyHumanResponseSchema]] = None,
        stream_mode: Literal["chunk", "message"] = "chunk"
    ) -> AsyncGenerator[BaseEvent, None]:
        ...

