from .base_tool import BaseTool
from typing import List, Optional
from abc import ABC, abstractmethod

class BaseToolManager(ABC):

    # === 增 ===
    @abstractmethod
    def add_tool(self, tool: BaseTool) -> None:
        ...



    # === 删 ===
    @abstractmethod
    def remove_tool_by_id(self, id: str) -> Optional[BaseTool]:
        ...

    @abstractmethod
    def clear_tools(self) -> None:
        ...



    # === 改 ===
    @abstractmethod
    def update_tool_by_id(self, id: str, new_tool: BaseTool) -> bool:
        ...

    @abstractmethod
    def update_tools(self, new_tools: List[BaseTool]) -> None:
        ...



    # === 查 ===
    @abstractmethod
    def get_tools(self) -> List[BaseTool]:
        ...

    @abstractmethod
    def get_tool_count(self) -> int:
        ...

    @abstractmethod
    def get_tool_by_id(self, id: str) -> Optional[BaseTool]:
        ...
    