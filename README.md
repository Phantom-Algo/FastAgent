# FastAgent - 轻量级 Agent Framework
## 版本
V0.2.5

## Quick Start

```python
from fast_agent import Agent, Context, LLMConfig, UserMessage, AssistantMessageOutputEvent

async def main():
    # 模型配置（以 deepseek 为例）
    llm_config = LLMConfig(
        model_name="deepseek-reasoner",
        api_key="your-api-key",
        base_url="https://api.deepseek.com",
        provider="deepseek"
    )

    # 上下文配置
    context = Context(
        system_prompt="你是一个乐于助人的人工智能助手，协助用户解答问题和提供信息。"
    )

    # 初始化 Agent
    agent = Agent(
        llm_config=llm_config,
        context=context
    )

    # 构造用户输入
    user_message = UserMessage(content="请告诉我今天的天气预报。")

    # 输出
    async for event in agent.stream(
        user_message,
        stream_mode="message"
    ):
        if isinstance(event, AssistantMessageOutputEvent):
            print(f"Assistant: {event.data.content}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Embeddings

```python
from fast_agent import AdapterFactory, EmbeddingConfig

async def main():
    embedding_config = EmbeddingConfig(
        model_name="text-embedding-3-small",
        api_key="your-api-key",
        base_url="https://api.openai.com/v1",
        provider="openai",
    )

    adapter = AdapterFactory.get_adapter_cls(embedding_config.provider)()
    response = await adapter.embed(
        embedding_config,
        ["FastAgent", "Embeddings module"],
    )

    print(response.model)
    print(len(response.data[0].embedding))
```