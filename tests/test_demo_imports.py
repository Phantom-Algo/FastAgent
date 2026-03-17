import sys
sys.path.insert(0, 'src')

from fast_agent.resources.agent.agent import Agent
from fast_agent.resources.agent.lifespan.lifespan_manager import LifespanManager
from fast_agent.resources.agent.lifespan.default_lifespan import DefaultExecutingTools
from fast_agent.resources.agent.event.events import (
    AskHumanEvent, AskHumanResponseEvent,
    AssistantMessageChunkOutputEvent, AssistantMessageOutputEvent,
    GuardTriggeredEvent, InterruptEvent, RoundStopEvent,
    ToolCallEvent, ToolsExecutedEvent,
)
from fast_agent.resources.agent.snapshot.snapshot import Snapshot
from fast_agent.resources.context.context import Context
from fast_agent.resources.llm.llm_config import LLMConfig
from fast_agent.resources.tool.tool_creator import tool_creator
from fast_agent.types.agent.lifespan.base_lifespan import IExecutingTools
from fast_agent.types.agent.lifespan.dto.lifespan_dto import ExecutingToolsRequest, ExecutingToolsResponse
from fast_agent.types.messages.domain.user_message import UserMessage
from fast_agent.types.tool.base_tool_runtime import BaseToolRuntime
from fast_agent.types.tool.domain.ask_human_policy import AskHumanPolicy
from fast_agent.types.tool.domain.guard_policy import GuardPolicy, GuardPolicyHumanResponseSchema

print("All imports OK")

# Test tool creation
@tool_creator(
    tool_name="test_tool",
    tool_description="A test tool",
)
async def test_tool(x: str) -> str:
    return f"result: {x}"

print(f"Tool created: {test_tool}")
print(f"Tool schema: {test_tool.to_openai_schema()}")

# Test guard tool
class TestApproval(GuardPolicyHumanResponseSchema):
    approved: bool = False

@tool_creator(
    tool_name="guard_test",
    tool_description="Guard test tool",
    guard_policy=GuardPolicy(
        info="Test guard",
        schema=TestApproval(),
        guard_func=lambda resp: getattr(resp, "approved", False),
    ),
)
async def guard_test(action: str) -> str:
    return f"executed: {action}"

print(f"Guard tool created: {guard_test}")
print(f"Guard policy: {guard_test.guard_policy}")

# Test snapshot serialization
from fast_agent.types.agent.fsm.enum.agent_fsm_state_enum import AgentFSMStateEnum
snapshot = Snapshot(
    llm_config=LLMConfig(model_name="test", api_key="test", base_url="test", provider="deepseek"),
    context=Context(system_prompt="test"),
    lifespan_manager=LifespanManager(),
    state=AgentFSMStateEnum.AFTER_USER_INPUT,
)
serialized = snapshot.serialize()
print(f"Snapshot serialized: {len(serialized)} bytes")
restored = Snapshot.deserialize(serialized)
print(f"Snapshot deserialized: ID={restored.id}, State={restored.state}")

print("\nAll tests passed!")
