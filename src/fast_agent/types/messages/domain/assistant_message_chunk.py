from ..base_chunk import BaseChunk
from typing import Optional, Literal

class AssistantMessageChunk(BaseChunk):
    """
    AssistantMessageChunk 大模型输出消息块类
    """
    type: Literal['assistant_message_chunk'] = 'assistant_message_chunk'

    reasoning_content_delta: Optional[str] = None

    content_delta: Optional[str] = None

    refusal_delta: Optional[str] = None