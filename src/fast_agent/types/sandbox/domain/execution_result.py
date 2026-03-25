from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from typing import Dict, List, Optional


class OutputMessage(BaseModel):
    """Single output event from stdout/stderr."""

    text: str = Field(description="Output text")
    timestamp: int = Field(
        description="Unix timestamp in milliseconds when output was generated"
    )
    is_error: bool = Field(
        default=False, description="Whether this message comes from stderr"
    )

    model_config = ConfigDict(populate_by_name=True)


class SingleExecutionResult(BaseModel):
    """Single displayable execution result item."""

    text: Optional[str] = Field(default=None, description="UTF-8 encoded result text")
    timestamp: int = Field(
        description="Unix timestamp in milliseconds when result was produced"
    )
    extra_properties: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional structured result content in UTF-8 format",
        alias="extra_properties",
    )

    model_config = ConfigDict(populate_by_name=True)


class ExecutionError(BaseModel):
    """Error payload when code execution fails."""

    name: str = Field(description="Error type, e.g. SyntaxError or RuntimeError")
    value: str = Field(description="Error message")
    timestamp: int = Field(
        description="Unix timestamp in milliseconds when error occurred"
    )
    traceback: List[str] = Field(default_factory=list, description="Error stack trace")

    model_config = ConfigDict(populate_by_name=True)


class ExecutionLogs(BaseModel):
    """Execution output stream container."""

    stdout: List[OutputMessage] = Field(
        default_factory=list, description="stdout output messages"
    )
    stderr: List[OutputMessage] = Field(
        default_factory=list, description="stderr output messages"
    )

    def add_stdout(self, message: OutputMessage) -> None:
        """Append one stdout output message."""
        self.stdout.append(message)

    def add_stderr(self, message: OutputMessage) -> None:
        """Append one stderr output message."""
        self.stderr.append(message)


class ExecutionComplete(BaseModel):
    """Execution completion event."""

    timestamp: int = Field(description="Unix timestamp in milliseconds when done")
    execution_time_in_millis: int = Field(
        description="Execution time in milliseconds", alias="execution_time_in_millis"
    )

    model_config = ConfigDict(populate_by_name=True)


class ExecutionInit(BaseModel):
    """Execution initialization event."""

    id: str = Field(description="Execution identifier")
    timestamp: int = Field(
        description="Unix timestamp in milliseconds when execution starts"
    )

    model_config = ConfigDict(populate_by_name=True)


class ExecutionResult(BaseModel):
    """
    Complete execution session model.

    This aggregate tracks the full lifecycle for one execution, including
    results, error details, and stdout/stderr logs.
    """

    id: Optional[str] = Field(default=None, description="Unique execution identifier")

    execution_count: Optional[int] = Field(
        default=None,
        description="Sequential execution counter",
        alias="execution_count",
    )

    result: List[SingleExecutionResult] = Field(
        default_factory=list, description="Execution results"
    )

    error: Optional[ExecutionError] = Field(
        default=None, description="Error info when execution fails"
    )

    logs: ExecutionLogs = Field(
        default_factory=ExecutionLogs, description="Captured output logs"
    )

    def add_result(self, result: SingleExecutionResult) -> None:
        """Append one execution result item."""
        self.result.append(result)

    @property
    def ok(self) -> bool:
        """Whether execution completed without an error payload."""
        return self.error is None

    model_config = ConfigDict(populate_by_name=True)