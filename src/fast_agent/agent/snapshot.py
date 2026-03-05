from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict
from fast_agent.llm import LLMConfig, Context, ToolCall, AssistantMessage, UserMessage, ToolResultMessage
from .state import AgentState
from typing import Optional, List, TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from .lifespan import Lifespan


class Snapshot(BaseModel):
    """
    Snapshot 快照类，用于保存和恢复 Agent 的状态。

    包含字段：
    - llm_config: 大模型配置
    - context: 上下文（检查点时刻的干净副本）
    - lifespan: 生命周期注册信息
    - user_input: 当前轮次的用户输入
    - llm_output: 最近一次 LLM 输出
    - tool_results: 最近一次工具执行结果
    - status: 中断时所处的状态阶段
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: f"snapshot_{uuid.uuid4().hex[:16]}")
    llm_config: LLMConfig
    context: Context
    lifespan: "Lifespan"
    user_input: Optional[UserMessage] = None
    llm_output: Optional[AssistantMessage] = None
    tool_results: Optional[List[ToolResultMessage]] = None
    status: AgentState