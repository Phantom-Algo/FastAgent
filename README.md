# FastAgent - 轻量级 Agent Framework

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