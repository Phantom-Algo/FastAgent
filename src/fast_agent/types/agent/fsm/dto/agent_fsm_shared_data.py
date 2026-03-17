from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, List
from ....llm.base_llm_config import BaseLLMConfig
from ....context.base_context import BaseContext
from ...lifespan.base_lifespan_manager import BaseLifespanManager
from ....messages.domain.assistant_message import AssistantMessage, ToolCall
from ....messages.domain.user_message import UserMessage
from ....messages.domain.tool_result_message import ToolResultMessage
from ....tool.domain.guard_policy import GuardPolicyHumanResponseSchema
from ...event.base_event_channel import BaseEventChannel

class AgentFSMSharedData(BaseModel):
    """AgentFSMSharedData 定义状态机状态变换过程中的共享数据"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm_config: BaseLLMConfig

    context: BaseContext

    lifespan_manager: BaseLifespanManager

    event_channel: BaseEventChannel

    llm_output: Optional[AssistantMessage] = None

    user_input: Optional[UserMessage] = None

    tool_results: Optional[List[ToolResultMessage]] = None

    human_response: Optional[Dict[str, GuardPolicyHumanResponseSchema]] = None

    pending_tool_calls: Optional[List[ToolCall]] = None

    finished_tool_calls: Optional[List[ToolCall]] = None

    prebuilt_tool_results: Optional[List[ToolResultMessage]] = None





