from pydantic import BaseModel, model_validator
from typing import Literal, Optional

class BasePart(BaseModel):
    """
    BasePart 基础消息部分类
    """
    type: str


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