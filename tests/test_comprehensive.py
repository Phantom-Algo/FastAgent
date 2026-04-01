"""
FastAgent 综合功能验证测试（不需要 API Key）

测试内容：
1. DeepSeek Adapter 注册和 payload 构建
2. 工具创建和 schema 生成
3. Guard 机制的工具检测
4. Snapshot 序列化/反序列化
5. 事件系统
6. Context 和 MessageManager
7. LifespanManager 自定义
"""

import sys
import os
import asyncio
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fast_agent.resources.adapter.adapter_factory import AdapterFactory
from fast_agent.resources.adapter.already_adapter.deepseek_adapter import DeepSeekAdapter
from fast_agent.resources.adapter.already_adapter.openai_adapter import OpenAIAdapter
from fast_agent.resources.agent.agent import Agent
from fast_agent.resources.agent.lifespan.lifespan_manager import LifespanManager
from fast_agent.resources.agent.lifespan.default_lifespan import (
    DefaultBeforeExecuteTools,
    DefaultExecutingTools,
)
from fast_agent.resources.agent.event.events import (
    AskHumanEvent,
    AskHumanResponseEvent,
    AssistantMessageChunkOutputEvent,
    AssistantMessageOutputEvent,
    GuardTriggeredEvent,
    InterruptEvent,
    RoundStopEvent,
    ToolCallEvent,
    ToolsExecutedEvent,
)
from fast_agent.resources.agent.snapshot.snapshot import Snapshot
from fast_agent.resources.context.context import Context
from fast_agent.resources.llm.llm_config import LLMConfig
from fast_agent.resources.tool.tool_creator import tool_creator
from fast_agent.resources.tool.tool_manager import ToolManager
from fast_agent.types.agent.fsm.enum.agent_fsm_state_enum import AgentFSMStateEnum
from fast_agent.types.agent.lifespan.dto.lifespan_dto import (
    BeforeExecuteToolsRequest,
    ExecutingToolsRequest,
)
from fast_agent.resources.agent.event.event_channel import EventChannel
from fast_agent.types.messages.domain.assistant_message import AssistantMessage, ToolCall
from fast_agent.types.messages.domain.tool_result_message import ToolResultMessage
from fast_agent.types.messages.domain.user_message import UserMessage
from fast_agent.types.tool.base_tool_runtime import BaseToolRuntime
from fast_agent.types.tool.domain.ask_human_policy import AskHumanPolicy
from fast_agent.types.tool.domain.guard_policy import GuardPolicy, GuardPolicyHumanResponseSchema
from fast_agent.types.tool.domain.guard_triggered import ToolCallGuardTriggeredException


passed = 0
failed = 0

def test(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}")


print("=" * 60)
print("FastAgent 综合功能验证测试")
print("=" * 60)

# ================================================================
# 1. DeepSeek Adapter
# ================================================================
print("\n📦 1. DeepSeek Adapter 测试")

test("DeepSeek adapter 已注册", AdapterFactory.get_adapter_cls("deepseek") == DeepSeekAdapter)
test("OpenAI adapter 仍可用", AdapterFactory.get_adapter_cls("openai") == OpenAIAdapter)

# 测试 DeepSeek payload 构建（不包含 parallel_tool_calls）
ds_adapter = DeepSeekAdapter()
llm_config = LLMConfig(
    model_name="deepseek-chat",
    api_key="test-key",
    base_url="https://api.deepseek.com",
    provider="deepseek",
)

@tool_creator(tool_name="test_tool", tool_description="Test tool")
async def _test_tool(x: str) -> str:
    return x

ctx = Context(system_prompt="test", tool_manager=[_test_tool])
payload = ds_adapter._build_chat_completion_payload(llm_config, ctx, stream=True)
test("DeepSeek payload 不含 parallel_tool_calls", "parallel_tool_calls" not in payload)
test("DeepSeek payload 含 tools", "tools" in payload)
test("DeepSeek payload model 正确", payload["model"] == "deepseek-chat")

# 测试 reasoning_content 保留
msg = AssistantMessage(
    reasoning_content="这是思维链内容",
    content="回答",
    finish_reason="stop",
)
converted = ds_adapter._convert_assistant_message(msg)
test("DeepSeek assistant message 保留 reasoning_content", converted.get("reasoning_content") == "这是思维链内容")

# OpenAI adapter 不保留 reasoning_content
oa_adapter = OpenAIAdapter()
oa_converted = oa_adapter._convert_assistant_message(msg)
test("OpenAI assistant message 无 reasoning_content", "reasoning_content" not in oa_converted)

# ================================================================
# 2. 工具创建
# ================================================================
print("\n🔧 2. 工具系统测试")

# 普通工具
@tool_creator(
    tool_name="get_weather",
    tool_description="查询天气",
)
async def get_weather(city: str) -> str:
    return f"天气: {city} 晴"

test("普通工具创建成功", get_weather.name == "get_weather")
test("普通工具无 guard_policy", get_weather.guard_policy is None)
test("普通工具有默认 ask_human_policy", get_weather.ask_human_policy is not None)

# Ask Human 工具
@tool_creator(
    tool_name="book_hotel",
    tool_description="预订酒店",
    ask_human_policy=AskHumanPolicy(timeout=120),
)
async def book_hotel(city: str, tool_runtime: BaseToolRuntime = None) -> str:
    return f"预订: {city}"

test("Ask Human 工具创建成功", book_hotel.name == "book_hotel")
test("Ask Human 工具有 tool_runtime_param_name", book_hotel.tool_runtime_param_name == "tool_runtime")
test("Ask Human 工具 timeout=120", book_hotel.ask_human_policy.timeout == 120)

# Guard 工具
class PaymentApproval(GuardPolicyHumanResponseSchema):
    approved: bool = False

def process_payment_guard_func(response: PaymentApproval) -> bool:
    return getattr(response, "approved", False)

@tool_creator(
    tool_name="process_payment",
    tool_description="处理支付",
    guard_policy=GuardPolicy(
        info="需要审批",
        schema=PaymentApproval(),
        guard_func=process_payment_guard_func,
    ),
)
async def process_payment(amount: str) -> str:
    return f"支付: {amount}"

test("Guard 工具创建成功", process_payment.name == "process_payment")
test("Guard 工具有 guard_policy", process_payment.guard_policy is not None)
test("Guard policy info 正确", process_payment.guard_policy.info == "需要审批")

# OpenAI schema
schema = get_weather.to_openai_schema()
test("OpenAI schema type=function", schema["type"] == "function")
test("OpenAI schema function name 正确", schema["function"]["name"] == "get_weather")

# ================================================================
# 3. Guard 机制检测
# ================================================================
print("\n🛡️ 3. Guard 机制测试")

tools = [get_weather, process_payment]
guard_ctx = Context(system_prompt="test", tool_manager=tools)

# 模拟 LLM 输出包含 guard 工具调用
tool_call_guard = ToolCall(
    tool_call_id="call_001",
    function_name="process_payment",
    function_args={"amount": "¥500"},
)
tool_call_normal = ToolCall(
    tool_call_id="call_002",
    function_name="get_weather",
    function_args={"city": "北京"},
)

llm_output_with_guard = AssistantMessage(
    content="让我处理",
    tool_calls=[tool_call_guard, tool_call_normal],
    finish_reason="tool_calls",
)

# 测试 Guard 检测（无 human_response 应触发异常）
async def test_guard_detection():
    handler = DefaultBeforeExecuteTools()
    event_channel = EventChannel()
    request = BeforeExecuteToolsRequest(
        llm_config=llm_config,
        context=guard_ctx,
        event_channel=event_channel,
        llm_output=llm_output_with_guard,
        kwargs={},
    )
    try:
        await handler.execute(request)
        return False, "应该抛出异常但没有"
    except ToolCallGuardTriggeredException as e:
        return True, f"捕获到 Guard 异常, {len(e.contexts)} 个触发"

guard_result, guard_msg = asyncio.run(test_guard_detection())
test(f"Guard 检测触发异常 ({guard_msg})", guard_result)

# 测试 Guard 通过（提供 human_response 且 approved=True）
async def test_guard_pass():
    handler = DefaultBeforeExecuteTools()
    event_channel = EventChannel()
    request = BeforeExecuteToolsRequest(
        llm_config=llm_config,
        context=guard_ctx,
        event_channel=event_channel,
        llm_output=llm_output_with_guard,
        kwargs={},
        human_response={"call_001": PaymentApproval(approved=True)},
    )
    response = await handler.execute(request)
    return len(response.pending_tool_calls) == 2

test("Guard 审批通过后工具可执行", asyncio.run(test_guard_pass()))

# 测试 Guard 拒绝（提供 human_response 且 approved=False）
async def test_guard_reject():
    handler = DefaultBeforeExecuteTools()
    event_channel = EventChannel()
    request = BeforeExecuteToolsRequest(
        llm_config=llm_config,
        context=guard_ctx,
        event_channel=event_channel,
        llm_output=llm_output_with_guard,
        kwargs={},
        human_response={"call_001": PaymentApproval(approved=False)},
    )
    response = await handler.execute(request)
    has_reject = any("拒绝" in str(r.content) or "rejected" in str(r.content) for r in response.prebuilt_tool_results)
    pending_count = len(response.pending_tool_calls)
    return has_reject and pending_count == 1

test("Guard 拒绝后生成拒绝结果", asyncio.run(test_guard_reject()))

# ================================================================
# 4. Snapshot 序列化/反序列化
# ================================================================
print("\n📸 4. Snapshot 序列化测试")

snapshot = Snapshot(
    llm_config=LLMConfig(
        model_name="deepseek-chat",
        api_key="test-key",
        base_url="https://api.deepseek.com",
        provider="deepseek",
    ),
    context=Context(system_prompt="你是一个旅行助手", tool_manager=[get_weather]),
    lifespan_manager=LifespanManager(),
    user_input=UserMessage(content="查询北京天气"),
    llm_output=AssistantMessage(
        reasoning_content="用户想查天气",
        content="让我查询",
        tool_calls=[ToolCall(function_name="get_weather", function_args={"city": "北京"})],
        finish_reason="tool_calls",
    ),
    state=AgentFSMStateEnum.BEFORE_EXECUTE_TOOLS,
)

serialized = snapshot.serialize()
test(f"序列化成功 ({len(serialized)} bytes)", len(serialized) > 0)

restored = Snapshot.deserialize(serialized)
test("反序列化 ID 一致", restored.id == snapshot.id)
test("反序列化 state 一致", restored.state == snapshot.state)
test("反序列化 user_input 一致", restored.user_input.content == "查询北京天气")
test("反序列化 llm_output reasoning 一致", restored.llm_output.reasoning_content == "用户想查天气")
test("反序列化 tool_calls 保留", len(restored.llm_output.tool_calls) == 1)
test("反序列化 llm_config 一致", restored.llm_config.model_name == "deepseek-chat")

# 二次序列化验证
serialized2 = restored.serialize()
restored2 = Snapshot.deserialize(serialized2)
test("二次序列化/反序列化 ID 一致", restored2.id == snapshot.id)

# ================================================================
# 5. 事件系统
# ================================================================
print("\n📡 5. 事件系统测试")

chunk_event = AssistantMessageChunkOutputEvent(
    data=AssistantMessageChunkOutputEvent.AssistantMessageChunkOutputEventData(
        chunk_type="reasoning_content",
        reasoning_content="思考中...",
    )
)
test("Chunk 事件创建成功", chunk_event.type == "chunk_output_event")
test("Chunk 事件数据正确", chunk_event.data.reasoning_content == "思考中...")

tool_call_event = ToolCallEvent(
    data=ToolCallEvent.ToolCallEventData(
        tool_call_id="call_001",
        function_name="get_weather",
        function_args={"city": "北京"},
    )
)
test("ToolCall 事件创建成功", tool_call_event.type == "tool_call_event")

guard_event = GuardTriggeredEvent(
    data=GuardTriggeredEvent.GuardTriggeredEventData(
        guard_triggered_contexts=[],
        snapshot=snapshot,
    )
)
test("GuardTriggered 事件创建成功", guard_event.type == "guard_triggered_event")

round_stop_event = RoundStopEvent(
    data=RoundStopEvent.RoundStopEventData(finish_reason="stop")
)
test("RoundStop 事件创建成功", round_stop_event.type == "round_stop_event")

# ================================================================
# 6. Context 管理
# ================================================================
print("\n📋 6. Context 管理测试")

ctx = Context(
    system_prompt="你是助手",
    tool_manager=[get_weather, process_payment],
)
test("Context 创建成功", ctx is not None)
test("System prompt 正确", "你是助手" in ctx.get_system_prompt().get_system_prompt())
test("Tool manager 包含 2 个工具", len(ctx.get_tool_manager().get_tools()) == 2)

ctx.set_tool_inject_param("user_id", "U001")
test("工具注入参数设置成功", ctx.get_tool_inject_param("user_id") == "U001")

ctx.add_work_message(UserMessage(content="你好"))
test("Work message 添加成功", len(ctx.get_work_message_manager().get_messages()) == 1)

# ================================================================
# 7. LifespanManager 自定义
# ================================================================
print("\n⚙️ 7. LifespanManager 测试")

lm = LifespanManager()
test("LifespanManager 创建成功", lm is not None)
test("默认 after_user_input 存在", lm.get_lifespan("after_user_input") is not None)
test("默认 executing_tools 存在", lm.get_lifespan("executing_tools") is not None)

# 自定义 ExecutingTools
class CustomExecutingTools(DefaultExecutingTools):
    pass

lm2 = LifespanManager(executing_tools=CustomExecutingTools())
test("自定义 executing_tools 注入成功", isinstance(lm2.get_lifespan("executing_tools"), CustomExecutingTools))

# kwargs
lm.update_kwargs({"test_key": "test_value"})
test("kwargs 更新成功", lm.get_kwargs().get("test_key") == "test_value")

# ================================================================
# 8. 工具执行测试 (不需要 LLM)
# ================================================================
print("\n⚡ 8. 工具执行测试")

async def test_tool_execution():
    handler = DefaultExecutingTools()
    event_channel = EventChannel()
    
    tool_call = ToolCall(
        tool_call_id="call_exec_001",
        function_name="get_weather",
        function_args={"city": "上海"},
    )
    
    tools_ctx = Context(system_prompt="test", tool_manager=[get_weather])
    
    request = ExecutingToolsRequest(
        llm_config=llm_config,
        context=tools_ctx,
        event_channel=event_channel,
        llm_output=AssistantMessage(content="test", tool_calls=[tool_call], finish_reason="tool_calls"),
        kwargs={},
        pending_tool_calls=[tool_call],
    )
    
    response = await handler.execute(request)
    results = response.tool_results
    return len(results) == 1 and "上海" in str(results[0].content)

test("普通工具异步执行成功", asyncio.run(test_tool_execution()))

# ================================================================
# 汇总
# ================================================================
print("\n" + "=" * 60)
total = passed + failed
print(f"测试结果: {passed}/{total} 通过, {failed}/{total} 失败")
if failed == 0:
    print("🎉 所有测试通过！")
else:
    print(f"⚠️ 有 {failed} 个测试失败")
    sys.exit(1)
