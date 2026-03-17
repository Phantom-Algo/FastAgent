from ..messages.base_message import BaseMessage
from ..messages.base_message_manager import BaseMessageManager
from ..system_prompt.base_system_prompt import BaseSystemPrompt
from ..tool.base_tool_manager import BaseToolManager
from typing import List, Dict, Any
from abc import ABC, abstractmethod


class BaseContext(ABC):
    """
    BaseContext 上下文管理基类，用于存储与管理传输给 LLM 的上下文信息
    """

    @abstractmethod
    def get_system_prompt(self) -> BaseSystemPrompt:
        """获取系统提示词"""
        ...

    @abstractmethod
    def get_work_message_manager(self) -> BaseMessageManager:
        """获取工作消息管理器"""
        ...

    @abstractmethod
    def get_tool_manager(self) -> BaseToolManager:
        """获取工具管理器"""
        ...

    # ===== 工具参数注入 API =====
    @abstractmethod
    def get_tool_inject_params(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def get_tool_inject_param(self, key: str, default: Any = None) -> Any:
        ...

    @abstractmethod
    def set_tool_inject_param(self, key: str, value: Any) -> None:
        ...

    @abstractmethod
    def set_tool_inject_params(self, params: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def update_tool_inject_params(self, params: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def remove_tool_inject_param(self, key: str, default: Any = None) -> Any:
        ...

    @abstractmethod
    def clear_tool_inject_params(self) -> None:
        ...

    @abstractmethod
    def has_tool_inject_param(self, key: str) -> bool:
        ...

    # ===== messages 管理 API =====
    @abstractmethod
    def add_raw_message(self, message: BaseMessage) -> None:
        ...

    @abstractmethod
    def add_raw_messages(self, messages: List[BaseMessage]) -> None:
        ...

    @abstractmethod
    def add_work_message(self, message: BaseMessage) -> None:
        ...

    @abstractmethod
    def add_work_messages(self, messages: List[BaseMessage]) -> None:
        ...