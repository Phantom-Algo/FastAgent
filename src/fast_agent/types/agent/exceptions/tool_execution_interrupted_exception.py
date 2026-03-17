from __future__ import annotations

from typing import List

from ...messages.domain.assistant_message import ToolCall
from ...messages.domain.tool_result_message import ToolResultMessage


class ToolExecutionInterruptedException(Exception):
    """Raised when tool execution partially succeeds and must be interrupted for snapshot-based resume."""

    def __init__(
        self,
        message: str,
        pending_tool_calls: List[ToolCall],
        finished_tool_calls: List[ToolCall],
        tool_results: List[ToolResultMessage],
    ):
        super().__init__(message)
        self.pending_tool_calls = pending_tool_calls
        self.finished_tool_calls = finished_tool_calls
        self.tool_results = tool_results
