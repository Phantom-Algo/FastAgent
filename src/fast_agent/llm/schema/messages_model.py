from pydantic import BaseModel, model_validator, Field
from typing import Literal, Optional, Union, List, Dict, Any
import uuid

class BaseMessage(BaseModel):
    """BaseMessage 基础消息类"""
    pass

# ====== UserMessage ======

class BasePart(BaseModel):
    """
    BasePart 基础消息部分类
    """
    pass

class TextPart(BasePart):
    """
    TextPart 文本消息部分类
    """
    type: Literal['text'] = 'text'
    text: str

class ImagePart(BasePart):
    """
    ImagePart 图片消息部分类
    """
    type: Literal['image'] = 'image'
    
    # Base64 编码（OpenAI Anthropic Google 等API风格均通用）
    base64_data: Optional[str] = None
    mime_type: Optional[str] = None

    # URL 网络地址（OpenAI API风格）
    url: Optional[str] = None

    # file_url 云端文件地址（Google API风格）
    file_url: Optional[str] = None

    # 清晰度（OpenAI API风格特殊字段）
    detail: Optional[Literal['auto', 'low', 'high']] = "auto"

    @model_validator(mode='after')
    def check_data_source(self):
        if not any([self.base64_data, self.url, self.file_url]):
            raise ValueError("ImagePart must have at least one data source: base64_data, url, or file_url.")
        if self.base64_data and not self.mime_type:
            raise ValueError("mime_type is required when base64_data is provided.")
        return self


class UserMessage(BaseMessage):
    """UserMessage 用户消息类"""
    role: Literal['user'] = 'user'

    content: Union[str, List[BasePart]]

# ====== AssistantMessage ======

class ToolCall(BaseModel):
    """ToolCall 大模型工具调用请求类"""
    # 工具调用唯一 ID，用于与后续的工具调用结果进行关联（为兼容无ID的情况，设置自动生成）
    id: str = Field(default_factory=lambda: f"call_{str(uuid.uuid4().hex[:16])}")

    type: Literal['tool_call'] = 'tool_call'

    function_name: str
    function_args: Dict[str, Any]

class AssistantMessage(BaseMessage):
    """AssistantMessage AI 消息类"""
    role: Literal['assistant'] = 'assistant'

    reasoning_content: Optional[str] = None

    content: Optional[str] = None

    tool_calls: Optional[List[ToolCall]] = None

    refusal: Optional[str] = None

    @model_validator(mode='after')
    def check_content_or_tools(self):
        """确保消息中至少有文本或者工具调用之一，避免空消息"""
        if not self.reasoning_content and not self.content and not self.tool_calls and not self.refusal:
            raise ValueError("AssistantMessage must contain 'reasoning_content', 'content', 'tool_calls', or 'refusal'")
        return self
    
# ====== ToolResultMessage ======

class ToolResultMessage(BaseMessage):
    """ToolResultMessage 工具调用结果消息类"""
    role: Literal['tool_result'] = 'tool_result'

    tool_call_id: str

    name: str

    content: Any

    is_error: bool = False


    



