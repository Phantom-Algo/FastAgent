from ..constant import DEFAULT_SYSTEM_PROMPT_CHIP_KEY, DEFAULT_SYSTEM_PROMPT_JSON_SCHEMA_CHIP_KEY
from .domain.system_prompt_chip import SystemPromptChipSchema, SystemPromptChipsSchema
from typing import Union, Optional, List, Literal
from pydantic import BaseModel
from enum import Enum
from abc import ABC, abstractmethod

class BaseSystemPrompt(ABC):

    class PromptType(Enum):
        """PromptType 枚举类，用于定义获取的系统提示词表现类型"""
        STR = "str"
        XML = "xml"

    
    @abstractmethod
    def get_system_prompt(self, type: Optional[Union[PromptType, Literal["str", "xml"]]] = None) -> str:
        ...
        
    
    # =================
    # chips API
    # =================

    @abstractmethod
    def add(self, key: str, content: Union[str, SystemPromptChipSchema, dict]) -> SystemPromptChipSchema:
        ...

    @abstractmethod
    def add_json_schema(self, schema: BaseModel, pre_prompt: Optional[str] = None, key: str = DEFAULT_SYSTEM_PROMPT_JSON_SCHEMA_CHIP_KEY) -> SystemPromptChipSchema:
        ...

    @abstractmethod
    def insert(self, key: str, content: Union[str, SystemPromptChipSchema, dict], index: int) -> SystemPromptChipSchema:
        ...

    @abstractmethod
    def move(self, key: str, index: int) -> None:
        ...

    @abstractmethod
    def remove(self, key: str) -> bool:
        ...

    @abstractmethod
    def ignore(self, key: str) -> str:
        ...

    @abstractmethod
    def wakeup(self, key: str) -> str:
        ...

    @abstractmethod
    def wakeup_all(self) -> List[str]:
        ...

    @abstractmethod
    def toggle(self, key: str) -> str:
        ...

    @abstractmethod
    def replace_chips(self, content: Union[str, SystemPromptChipsSchema, dict]) -> None:
        ...

    @abstractmethod
    def update(self, content: Union[str, SystemPromptChipSchema, dict], key: str = DEFAULT_SYSTEM_PROMPT_CHIP_KEY) -> None:
        ...

    @abstractmethod
    def get(self, key: str) -> Optional[SystemPromptChipSchema]:
        ...

    @abstractmethod
    def get_chips(self) -> SystemPromptChipsSchema:
        ...