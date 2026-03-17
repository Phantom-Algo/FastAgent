from ...types.tool.base_tool_manager import BaseToolManager
from ...types.tool.base_tool import BaseTool
from typing import List, Optional


class ToolManager(BaseToolManager):

    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None
    ):
        self.tools = tools if tools is not None else []

    # === 增 ===
    def add_tool(self, tool: BaseTool) -> None:
        self.tools.append(tool)

    # === 删 ===
    def remove_tool_by_id(self, id: str) -> Optional[BaseTool]:
        for index, tool in enumerate(self.tools):
            if tool.id == id:
                return self.tools.pop(index)
        return None

    def clear_tools(self) -> None:
        self.tools.clear()

    # === 改 ===
    def update_tool_by_id(self, id: str, new_tool: BaseTool) -> bool:
        for index, tool in enumerate(self.tools):
            if tool.id == id:
                self.tools[index] = new_tool.model_copy(update={"id": id})
                return True
        return False

    def update_tools(self, new_tools: List[BaseTool]) -> None:
        self.tools = list(new_tools)

    # === 查 ===
    def get_tools(self) -> List[BaseTool]:
        return list(self.tools)

    def get_tool_count(self) -> int:
        return len(self.tools)

    def get_tool_by_id(self, id: str) -> Optional[BaseTool]:
        for tool in self.tools:
            if tool.id == id:
                return tool
        return None

    