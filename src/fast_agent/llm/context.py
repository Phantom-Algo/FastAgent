from pydantic import BaseModel
from typing import Optional, Union, List
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
        tools: Optional[Union[Tools, List[BaseTool]]] = None
    ):
        # 初始化 system_prompt
        if isinstance(system_prompt, str):
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


        # 初始化 subsquent_messages
        self.subsquent_messages: List[BaseMessage] = []


        # 初始化 tools
        if tools is None:
            pass
        elif isinstance(tools, list):
            pass
        elif isinstance(tools, Tools):
            self.tools = tools
        else:
            raise ValueError("tools must be a list of BaseTool or Tools instance")
