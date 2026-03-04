from typing import Optional, Union, List, Dict, Any
from .schema import SystemPrompt, Messages, Tools, BaseMessage
from fast_agent.tool import BaseTool
from copy import deepcopy

class Context:
    """
    Context 上下文类，用于存储与管理传输给 LLM 的上下文信息，包含以下字段：

    - system_prompt：系统提示词

    - work_messages：工作区消息列表

    - raw_messages：原始消息列表

    - tools：工具列表
    """

    def __init__(
        self,
        system_prompt: Optional[Union[SystemPrompt, str]] = None,
        work_messages: Optional[Union[Messages, List[BaseMessage]]] = None,
        tools: Optional[Union[Tools, List[BaseTool]]] = None,
        tool_inject_params: Optional[Dict[str, Any]] = None
    ):
        # 初始化 system_prompt
        if system_prompt is None:
            self.system_prompt = SystemPrompt(content="")
        elif isinstance(system_prompt, str):
            self.system_prompt = SystemPrompt(content=system_prompt)
        elif isinstance(system_prompt, SystemPrompt):
            self.system_prompt = system_prompt
        else:
            raise ValueError("system_prompt must be a string or SystemPrompt instance")
        

        # 初始化 work_messages
        if work_messages is None:
            self.work_messages = Messages(messages=[])
        elif isinstance(work_messages, list):
            self.work_messages = Messages(messages=work_messages)
        elif isinstance(work_messages, Messages):
            self.work_messages = work_messages
        else:
            raise ValueError("work_messages must be a list of BaseMessage or Messages instance")
        

        # 初始化 raw_messages
        self.raw_messages = deepcopy(self.work_messages)


        # 初始化 subsequent_messages
        self.subsequent_messages: List[BaseMessage] = []


        # 初始化 tools
        if tools is None:
            self.tools = Tools(tools=[])
        elif isinstance(tools, list):
            self.tools = Tools(tools=tools)
        elif isinstance(tools, Tools):
            self.tools = tools
        else:
            raise ValueError("tools must be a list of BaseTool or Tools instance")

        # 初始化 tool_inject_params
        self.tool_inject_params = tool_inject_params or {}

    
    # ===== 工具参数注入 API =====
    def get_tool_inject_params(self) -> Dict[str, Any]:
        return self.tool_inject_params

    def get_tool_inject_param(self, key: str, default: Any = None) -> Any:
        return self.tool_inject_params.get(key, default)

    def set_tool_inject_param(self, key: str, value: Any) -> None:
        self.tool_inject_params[key] = value

    def set_tool_inject_params(self, params: Dict[str, Any]) -> None:
        self.tool_inject_params = dict(params)

    def update_tool_inject_params(self, params: Dict[str, Any]) -> None:
        self.tool_inject_params.update(params)

    def remove_tool_inject_param(self, key: str, default: Any = None) -> Any:
        return self.tool_inject_params.pop(key, default)

    def clear_tool_inject_params(self) -> None:
        self.tool_inject_params.clear()

    def has_tool_inject_param(self, key: str) -> bool:
        return key in self.tool_inject_params
    
    # ===== messages 管理 API =====
    def add_raw_message(self, message: BaseMessage) -> None:
        raw_message = deepcopy(message)
        self.raw_messages.add_message(raw_message)
        self.subsequent_messages.append(message)

    def add_raw_messages(self, messages: List[BaseMessage]) -> None:
        for message in messages:
            self.add_raw_message(message)

    def add_work_message(self, message: BaseMessage) -> None:
        self.work_messages.add_message(message)

    def add_work_messages(self, messages: List[BaseMessage]) -> None:
        self.work_messages.add_messages(messages)
