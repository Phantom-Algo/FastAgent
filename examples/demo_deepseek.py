import os
import sys
from pathlib import Path
from copy import deepcopy

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fast_agent import Agent, Lifespan, LLMConfig, Context, tool
from fast_agent.llm import UserMessage, AssistantMessage
from fast_agent.agent import (
    AssistantMessageChunkOutputEvent,
    ToolCallEvent,
    AssistantMessageOutputEvent,
    ToolsExecutedEvent,
    RoundStopEvent,
)
from fast_agent.agent.lifespan import (
    IAfterLLMOutput,
    IAfterFinish,
    InputAfterLLMOutput,
    OutputAfterLLMOutput,
    InputAfterFinish,
    OutputAfterFinish,
)


@tool(tool_name="add_numbers", inject_params=["request_id"])
def add_numbers(a: int, b: int, request_id: str) -> str:
    return f"sum={a + b}, request_id={request_id}"


@tool(tool_name="get_weather")
def get_weather(city: str) -> str:
    city_normalized = city.strip() or "unknown"
    return f"{city_normalized} 今天天气晴，最高温 26°C（demo mock）"


class DeepSeekThinkingAfterLLMOutput(IAfterLLMOutput):
    async def after_llm_output(self, data: InputAfterLLMOutput) -> OutputAfterLLMOutput:
        llm_output = data.llm_output

        if llm_output is None:
            return OutputAfterLLMOutput(
                llm_config=data.llm_config,
                context=data.context,
                llm_output=llm_output,
                kwargs=data.kwargs,
            )

        if llm_output.finish_reason == "tool_calls":
            data.kwargs["had_tool_round"] = True

        # 关键策略：
        # - tool_calls 回合保留 reasoning_content（让下一轮模型可见）
        # - stop 回合先不在这里删，统一交给 after_finish 执行“轮次收尾清理”
        return OutputAfterLLMOutput(
            llm_config=data.llm_config,
            context=data.context,
            llm_output=llm_output,
            kwargs=data.kwargs,
        )


class DeepSeekThinkingAfterFinish(IAfterFinish):
    async def after_finish(self, data: InputAfterFinish) -> OutputAfterFinish:
        # 一轮完整交互结束后，清理 work_messages 中 assistant 的 reasoning_content
        # 避免把思考链长期留在后续轮次上下文里。
        cleaned_messages = []
        for message in data.context.work_messages.get_messages():
            if getattr(message, "role", None) == "assistant" and isinstance(message, AssistantMessage):
                cleaned_messages.append(
                    message.model_copy(update={"reasoning_content": None})
                )
            else:
                cleaned_messages.append(message)

        data.context.work_messages.update_messages(cleaned_messages)

        return OutputAfterFinish(
            llm_config=data.llm_config,
            context=data.context,
            kwargs=data.kwargs,
        )


async def main() -> None:
    load_dotenv(ROOT / "examples" / ".env")

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError("请先在 examples/.env 中设置 DEEPSEEK_API_KEY")

    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    llm_config = LLMConfig(
        provider="deepseek",
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.2,
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    context = Context(
        system_prompt="你是一个乐于助人的助手。先思考再回答；需要时调用工具。",
        tools=[add_numbers, get_weather],
        tool_inject_params={"request_id": "req-demo-001"},
    )

    lifespan = Lifespan(
        after_llm_output=DeepSeekThinkingAfterLLMOutput(),
        after_finish=DeepSeekThinkingAfterFinish(),
        kwargs={"demo": "deepseek-thinking"},
    )

    agent = Agent(llm_config=llm_config, context=context, lifespan=lifespan)

    user_input = UserMessage(
        content="请先计算 12+30，再查询北京天气，最后给我一个简短总结。"
    )

    async for event in agent.stream(user_input=user_input):
        if isinstance(event, AssistantMessageChunkOutputEvent):
            if event.data.type == "reasoning_content" and event.data.reasoning_content:
                print(event.data.reasoning_content, end="", flush=True)
            elif event.data.type == "content" and event.data.content:
                print(event.data.content, end="", flush=True)
            elif event.data.type == "refusal" and event.data.refusal:
                print(event.data.refusal, end="", flush=True)

        elif isinstance(event, ToolCallEvent):
            print(f"\n[ToolCall] {event.data.function_name} args={event.data.function_args}")

        elif isinstance(event, ToolsExecutedEvent):
            print("\n[ToolsExecuted]")
            for tool_result in event.data.tool_results:
                print(f"  - {tool_result.name}: {tool_result.content}")

        elif isinstance(event, AssistantMessageOutputEvent):
            print(f"\n[AssistantMessageOutput] finish_reason={event.data.finish_reason}")

        elif isinstance(event, RoundStopEvent):
            print(f"\n[RoundStop] finish_reason={event.data.finish_reason}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
