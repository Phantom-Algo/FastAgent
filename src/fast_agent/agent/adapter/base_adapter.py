from abc import ABC, abstractmethod
from fast_agent.llm import LLMConfig, Context, AssistantMessageChunk, ToolCall, AssistantMessage
from typing import AsyncGenerator, Union

class IAdapter(ABC):
    """
    IAdapter 规定了不同模型厂商适配器的标准接口 
    """
    @abstractmethod
    async def stream(self, llm_config: LLMConfig, context: Context) -> AsyncGenerator[Union[AssistantMessageChunk, ToolCall, AssistantMessage], None]:
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
    async def invoke(self, llm_config: LLMConfig, context: Context) -> AssistantMessage:
        """
        同步输出接口

        参数列表：
        - llm_config: LLMConfig 大模型配置
        - context: Context 上下文信息

        返回值：
        - AssistantMessage: 最终生成的消息对象
        """
        pass
        