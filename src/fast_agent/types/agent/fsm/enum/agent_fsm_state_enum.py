from enum import Enum

class AgentFSMStateEnum(Enum):
    """Agent FSM 状态枚举，表示状态机中的各个阶段"""
    AFTER_USER_INPUT = "after_user_input"
    LLM_OUTPUT = "llm_output"
    AFTER_LLM_OUTPUT = "after_llm_output"
    BEFORE_EXECUTE_TOOLS = "before_execute_tools"
    EXECUTING_TOOLS = "executing_tools"
    AFTER_EXECUTE_TOOLS = "after_execute_tools"
    AFTER_FINISH = "after_finish"