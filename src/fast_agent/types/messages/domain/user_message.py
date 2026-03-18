from pydantic import Field
from typing import Literal, Union, List
import uuid
from ..base_message import BaseMessage
from .content_part import BasePart


class UserMessage(BaseMessage):
    """UserMessage 用户消息类"""
    id: str = Field(default_factory=lambda: f"user_{str(uuid.uuid4().hex[:16])}")

    type: Literal['user_message'] = 'user_message'

    role: Literal['user'] = 'user'

    content: Union[str, List[BasePart]]