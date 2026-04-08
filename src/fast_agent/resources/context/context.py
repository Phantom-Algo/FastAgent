from ...types.context.base_context import BaseContext
from ...types.system_prompt.base_system_prompt import BaseSystemPrompt
from ...types.messages.base_message_manager import BaseMessageManager
from ...types.messages.base_message import BaseMessage
from ...types.tool.base_tool_manager import BaseToolManager
from ...types.tool.base_tool import BaseTool

from ..messages.message_manager import MessageManager
from ..tool.tool_manager import ToolManager
from ..system_prompt.system_prompt import SystemPrompt
from typing import Optional, Dict, Any, Union, List
from copy import deepcopy


def _safe_deepcopy_value(value: Any, memo: Optional[dict] = None) -> Any:
    try:
        return deepcopy(value, memo or {})
    except Exception:
        return value


def _safe_deepcopy_dict(data: Dict[str, Any], memo: Optional[dict] = None) -> Dict[str, Any]:
    copied: Dict[str, Any] = {}
    for key, value in data.items():
        copied_key = _safe_deepcopy_value(key, memo)
        copied[copied_key] = _safe_deepcopy_value(value, memo)
    return copied


class Context(BaseContext):

    def __init__(
        self,
        system_prompt: Optional[Union[str, BaseSystemPrompt]] = None,
        work_message_manager: Optional[Union[List[BaseMessage], BaseMessageManager]] = None,
        tool_manager: Optional[Union[List[BaseTool], BaseToolManager]] = None,
        tool_inject_params: Optional[Dict[str, Any]] = None,
    ):
        if system_prompt is None:
            self.system_prompt = SystemPrompt("")
        elif isinstance(system_prompt, str):
            self.system_prompt = SystemPrompt(system_prompt)
        elif isinstance(system_prompt, BaseSystemPrompt):
            self.system_prompt = system_prompt
        else:
            raise ValueError(
                "Error! Unsupported system_prompt type for Context. Expected None, str, or BaseSystemPrompt."
            )

        if work_message_manager is None:
            self.work_message_manager = MessageManager()
        elif isinstance(work_message_manager, list):
            self.work_message_manager = MessageManager(work_message_manager)
        elif isinstance(work_message_manager, BaseMessageManager):
            self.work_message_manager = work_message_manager
        else:
            raise ValueError(
                "Error! Unsupported work_message_manager type for Context. Expected None, list, or BaseMessageManager."
            )

        # raw_message_manager 保存初始化时的原始消息快照
        self.raw_message_manager = deepcopy(self.work_message_manager)

        # subsequent_message_manager 仅记录后续新增消息
        self.subsequent_message_manager = MessageManager()

        if tool_manager is None:
            self.tool_manager = ToolManager()
        elif isinstance(tool_manager, list):
            self.tool_manager = ToolManager(tool_manager)
        elif isinstance(tool_manager, BaseToolManager):
            self.tool_manager = tool_manager
        else:
            raise ValueError(
                "Error! Unsupported tool_manager type for Context. Expected None, list, or BaseToolManager."
            )

        self.tool_inject_params = dict(tool_inject_params or {})

    def __deepcopy__(self, memo):
        copied = self.__class__.__new__(self.__class__)
        memo[id(self)] = copied

        copied.system_prompt = deepcopy(self.system_prompt, memo)
        copied.work_message_manager = deepcopy(self.work_message_manager, memo)
        copied.raw_message_manager = deepcopy(self.raw_message_manager, memo)
        copied.subsequent_message_manager = deepcopy(self.subsequent_message_manager, memo)
        copied.tool_manager = deepcopy(self.tool_manager, memo)
        copied.tool_inject_params = _safe_deepcopy_dict(self.tool_inject_params, memo)
        return copied

    def get_system_prompt(self) -> BaseSystemPrompt:
        return self.system_prompt
    
    def get_work_message_manager(self) -> BaseMessageManager:
        return self.work_message_manager
    
    def get_tool_manager(self) -> BaseToolManager:
        return self.tool_manager

    # ===== 工具参数注入 API =====
    def get_tool_inject_params(self) -> Dict[str, Any]:
        return dict(self.tool_inject_params)

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
        deepcopy_message = deepcopy(message)
        self.raw_message_manager.add_message(deepcopy_message)
        self.subsequent_message_manager.add_message(deepcopy_message)

    def add_raw_messages(self, messages: List[BaseMessage]) -> None:
        deepcopy_messages = deepcopy(messages)
        self.raw_message_manager.add_messages(deepcopy_messages)
        self.subsequent_message_manager.add_messages(deepcopy_messages)

    def add_work_message(self, message: BaseMessage) -> None:
        self.work_message_manager.add_message(message)

    def add_work_messages(self, messages: List[BaseMessage]) -> None:
        self.work_message_manager.add_messages(messages)