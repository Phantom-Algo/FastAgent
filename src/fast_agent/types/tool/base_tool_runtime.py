from pydantic import BaseModel, ConfigDict
from .base_tool import BaseTool
from ..messages.domain.assistant_message import AssistantMessage
from ..messages.domain.user_message import UserMessage
from .base_ask_human_channel import BaseAskHumanChannel
from ..context.base_context import BaseContext
from ..llm.base_llm_config import BaseLLMConfig
from typing import Optional, Dict, Any

class BaseToolRuntime(BaseModel):
    """
    BaseToolRuntime 包含工具执行时的上下文信息，供工具函数使用。
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool_call_id: str

    this_tool: Optional[BaseTool] = None

    llm_config: Optional[BaseLLMConfig] = None

    context: Optional[BaseContext] = None

    llm_output: Optional[AssistantMessage] = None

    user_input: Optional[UserMessage] = None

    ask_human_channel: Optional[BaseAskHumanChannel] = None

    kwars: Optional[Dict[str, Any]] = None

    async def ask_human(self, data: Dict[str, Any], timeout: Optional[int] = None) -> Dict[str, Any]:
        ...