from ...types.tool.base_tool_runtime import BaseToolRuntime
from ...types.constant import DEFAULT_ASK_HUMAN_TIMEOUT
from typing import Dict, Any, Optional

class ToolRuntime(BaseToolRuntime):

    async def ask_human(self, data: Dict[str, Any], timeout: Optional[int] = None) -> Dict[str, Any]:
        if timeout is None:
            ask_human_policy = self.this_tool.ask_human_policy
            if ask_human_policy is not None:
                timeout = ask_human_policy.timeout
            else:
                timeout = DEFAULT_ASK_HUMAN_TIMEOUT  # 默认超时时间为 300 秒

        return await self.ask_human_channel.ask_human(
            data, 
            timeout
        )