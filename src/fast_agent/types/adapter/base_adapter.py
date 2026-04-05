from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Union

from ..llm.base_llm_config import BaseLLMConfig
from ..context.base_context import BaseContext
from ..embeddings.base_embedding_config import BaseEmbeddingConfig
from ..embeddings.domain import EmbeddingResponse
from ..messages.domain import AssistantMessage, ToolCall, AssistantMessageChunk

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

    @abstractmethod
    async def invoke(self, llm_config: BaseLLMConfig, context: BaseContext) -> AssistantMessage:
        """
        非流式输出接口

        参数列表：
        - llm_config: LLMConfig 大模型配置
        - context: Context 上下文信息

        返回值：
        - AssistantMessage: 最终的助手消息
        """
        pass

    @abstractmethod
    async def embed(self, embedding_config: BaseEmbeddingConfig, inputs: Union[str, List[str]]) -> EmbeddingResponse:
        """
        文本向量化接口

        参数列表：
        - embedding_config: EmbeddingConfig 向量模型配置
        - inputs: 单条文本或批量文本

        返回值：
        - EmbeddingResponse: 统一的向量化响应
        """
        pass