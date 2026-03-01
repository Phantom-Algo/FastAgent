from fast_agent.tool import BaseTool
from typing import List, Optional

class Tools:
    """
    Tools 工具类，用于定义与管理工具
    """
    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None
    ):
        if tools is None:
            self.tools = []
        elif isinstance(tools, list):
            self.tools = tools
        else:
            raise ValueError("tools must be a list of BaseTool instances")
        
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
                self.tools[index] = new_tool
                return True
        return False

    def update_tools(self, new_tools: List[BaseTool]) -> None:
        self.tools = new_tools


    # === 查 ===
    def get_tools(self) -> List[BaseTool]:
        return self.tools

    def get_tool_count(self) -> int:
        return len(self.tools)

    def get_tool_by_id(self, id: str) -> Optional[BaseTool]:
        for tool in self.tools:
            if tool.id == id:
                return tool
        return None