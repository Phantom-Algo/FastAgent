from abc import ABC, abstractmethod
from ..llm.base_llm_config import BaseLLMConfig
from ..context.base_context import BaseContext
from ..messages.domain import AssistantMessage, ToolCall, AssistantMessageChunk
from typing import AsyncGenerator, Union

class IAdapter(ABC):
    """
    IAdapter 规定了不同模型厂商适配器的标准接口 
    """
    @abstractmethod
    async def stream(self, llm_config: BaseLLMConfig, context: BaseContext) -> AsyncGenerator[Union[AssistantMessageChunk, ToolCall, AssistantMessage], None]:
        """
        流式输出接口

        参数列表：
        - llm_config: LLMConfig 大模型配置
        - context: Context 上下文信息

        返回值：
        - AsyncGenerator[Union[AssistantMessageChunk, ToolCall, AssistantMessage], None]: 流式输出内容的异步生成器，每次迭代返回一个消息块、工具调用或最终消息
        """
        pass