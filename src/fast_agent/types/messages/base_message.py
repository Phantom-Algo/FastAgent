from pydantic import BaseModel, Field
import uuid

class BaseMessage(BaseModel):
    """BaseMessage 基础消息类"""
    id: str = Field(default_factory=lambda: f"msg_{str(uuid.uuid4().hex[:16])}")

    type: str

    role: str