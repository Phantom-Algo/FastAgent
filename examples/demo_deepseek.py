import os
import sys
from pathlib import Path
from typing import Optional

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
    InterruptEvent,
    Snapshot,
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
    """演示工具：计算两数之和，并返回注入参数。"""
    return f"sum={a + b}, request_id={request_id}"


@tool(tool_name="get_weather")
def get_weather(city: str) -> str:
    """演示工具：返回模拟天气信息。"""
    city_normalized = city.strip() or "unknown"
    return f"{city_normalized} 今天天气晴，最高温 26°C（demo mock）"


class DeepSeekThinkingAfterLLMOutput(IAfterLLMOutput):
    """在工具回合保留 reasoning_content，便于下一轮可见。"""

    async def after_llm_output(self, data: InputAfterLLMOutput) -> OutputAfterLLMOutput:
        llm_output = data.llm_output

        if llm_output is not None and llm_output.finish_reason == "tool_calls":
            data.kwargs["had_tool_round"] = True

        return OutputAfterLLMOutput(
            llm_config=data.llm_config,
            context=data.context,
            llm_output=llm_output,
            kwargs=data.kwargs,
        )


class DeepSeekThinkingAfterFinish(IAfterFinish):
    """每轮结束后清理 work_messages 里的推理内容，避免长期污染上下文。"""

    async def after_finish(self, data: InputAfterFinish) -> OutputAfterFinish:
        cleaned_messages = []
        for message in data.context.work_messages.get_messages():
            if getattr(message, "role", None) == "assistant" and isinstance(message, AssistantMessage):
                cleaned_messages.append(message.model_copy(update={"reasoning_content": None}))
            else:
                cleaned_messages.append(message)

        data.context.work_messages.update_messages(cleaned_messages)

        return OutputAfterFinish(
            llm_config=data.llm_config,
            context=data.context,
            kwargs=data.kwargs,
        )


def build_agent() -> Agent:
    """构建 DeepSeek Agent。"""
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
        kwargs={"demo": "deepseek-thinking-snapshot"},
    )

    return Agent(llm_config=llm_config, context=context, lifespan=lifespan)


def _print_event(event, prefix: str = "") -> Optional[Snapshot]:
    """统一打印事件；若为中断事件则返回 snapshot。"""
    if isinstance(event, AssistantMessageChunkOutputEvent):
        if event.data.type == "reasoning_content" and event.data.reasoning_content:
            print(event.data.reasoning_content, end="", flush=True)
        elif event.data.type == "content" and event.data.content:
            print(event.data.content, end="", flush=True)
        elif event.data.type == "refusal" and event.data.refusal:
            print(event.data.refusal, end="", flush=True)

    elif isinstance(event, ToolCallEvent):
        print(f"\n{prefix}[ToolCall] {event.data.function_name} args={event.data.function_args}")

    elif isinstance(event, ToolsExecutedEvent):
        print(f"\n{prefix}[ToolsExecuted]")
        for tool_result in event.data.tool_results:
            print(f"{prefix}  - {tool_result.name}: {tool_result.content}")

    elif isinstance(event, AssistantMessageOutputEvent):
        print(f"\n{prefix}[AssistantMessageOutput] finish_reason={event.data.finish_reason}")

    elif isinstance(event, RoundStopEvent):
        print(f"\n{prefix}[RoundStop] finish_reason={event.data.finish_reason}")

    elif isinstance(event, InterruptEvent):
        print(f"\n{prefix}[Interrupt] reason={event.data.reason}")
        print(
            f"{prefix}[Snapshot] id={event.data.snapshot.id}, status={event.data.snapshot.status.value}"
        )
        return event.data.snapshot

    return None


async def run_with_mock_client_error(agent: Agent, user_input: UserMessage) -> Optional[Snapshot]:
    """
    第一阶段：模拟客户端异常并触发中断。

    模拟方式：
    - 当收到第一个工具调用事件时，模拟“客户端处理链路异常”；
    - 通过 request_interrupt 通知服务端中断当前流；
    - 等待 InterruptEvent，并提取 snapshot。
    """
    print("\n========== 第一阶段：模拟客户端出错并中断 ==========")

    snapshot: Optional[Snapshot] = None
    injected = False

    async for event in agent.stream(user_input=user_input):
        maybe_snapshot = _print_event(event)
        if maybe_snapshot is not None:
            snapshot = maybe_snapshot
            break

        if not injected and isinstance(event, ToolCallEvent):
            injected = True
            print("\n\n[MockClientError] 客户端处理事件时发生异常，准备发起中断请求。")
            agent.request_interrupt(reason="mock_client_error: consumer_pipeline_broken")

    return snapshot


async def run_resume_from_snapshot(agent: Agent, snapshot: Snapshot) -> None:
    """第二阶段：基于快照恢复，并继续完整输出。"""
    print("\n========== 第二阶段：从快照恢复并继续输出 ==========")

    async for event in agent.resume_stream(snapshot=snapshot):
        _print_event(event, prefix="[Resume] ")


async def main() -> None:
    agent = build_agent()
    user_input = UserMessage(content="请先计算 12+30，再查询北京天气，最后给我一个简短总结。")

    snapshot = await run_with_mock_client_error(agent=agent, user_input=user_input)
    if snapshot is None:
        raise RuntimeError("未获取到中断快照，无法演示恢复流程。")

    await run_resume_from_snapshot(agent=agent, snapshot=snapshot)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
