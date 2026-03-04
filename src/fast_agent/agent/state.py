"""
Agent 状态枚举模块

定义了 Agent 状态机中各个阶段的枚举值，供状态机、快照和外部使用。
"""

from enum import Enum


class AgentState(Enum):
    """Agent 状态枚举，表示状态机中的各个阶段"""
    AFTER_USER_INPUT = "after_user_input"
    LLM_OUTPUT = "llm_output"
    AFTER_LLM_OUTPUT = "after_llm_output"
    BEFORE_EXECUTE_TOOLS = "before_execute_tools"
    EXECUTING_TOOLS = "executing_tools"
    AFTER_EXECUTE_TOOLS = "after_execute_tools"
    AFTER_FINISH = "after_finish"
