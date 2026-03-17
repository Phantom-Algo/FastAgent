from enum import Enum


class LifespanType(str, Enum):
	"""生命周期阶段类型。"""

	AFTER_FINISH = "after_finish"
	AFTER_USER_INPUT = "after_user_input"
	AFTER_LLM_OUTPUT = "after_llm_output"
	BEFORE_EXECUTE_TOOLS = "before_execute_tools"
	EXECUTING_TOOLS = "executing_tools"
	AFTER_EXECUTE_TOOLS = "after_execute_tools"
