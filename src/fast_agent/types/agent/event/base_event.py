from pydantic import BaseModel, Field
from typing import Any
import time
import uuid

class BaseEventMetadata(BaseModel):
    """BaseEventMetadata 事件元数据基类"""
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))

class BaseEvent(BaseModel):
    """BaseEvent 事件基类"""
    id: str = Field(default_factory=lambda: f"event_{uuid.uuid4().hex[:16]}")

    type: str

    data: Any

    metadata: BaseEventMetadata = Field(default_factory=BaseEventMetadata)