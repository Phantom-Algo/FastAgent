from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List, Any
from ...llm.base_llm_config import BaseLLMConfig
from ...context.base_context import BaseContext
from ..lifespan.base_lifespan_manager import BaseLifespanManager
from ...messages.domain import UserMessage, AssistantMessage, ToolResultMessage, ToolCall
from ...tool.domain.guard_triggered import ToolCallGuardTriggeredContext
from ..fsm.enum.agent_fsm_state_enum import AgentFSMStateEnum
import uuid

class BaseSnapshot(BaseModel):
    """ BaseSnapshot 快照基类，用于保存和恢复 Agent 的状态。"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: f"snapshot_{uuid.uuid4().hex[:16]}")

    llm_config: BaseLLMConfig

    context: BaseContext

    lifespan_manager: BaseLifespanManager

    user_input: Optional[UserMessage] = None

    llm_output: Optional[AssistantMessage] = None

    tool_results: Optional[List[ToolResultMessage]] = None

    tool_call_guard_triggered_contexts: Optional[List[ToolCallGuardTriggeredContext]] = None

    pending_tool_calls: Optional[List[ToolCall]] = None

    finished_tool_calls: Optional[List[ToolCall]] = None

    prebuilt_tool_results: Optional[List[ToolResultMessage]] = None

    state: AgentFSMStateEnum


    def serialize(self) -> Any:
        ...


    @classmethod
    def deserialize(data: Any) -> "BaseSnapshot":
        ...





    

