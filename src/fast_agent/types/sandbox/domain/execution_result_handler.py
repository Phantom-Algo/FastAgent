from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field

from .execution_result import (
    ExecutionComplete,
    ExecutionError,
    ExecutionInit,
    OutputMessage,
    SingleExecutionResult,
)

Handler = Callable[[object], Awaitable[None]]

OutputHandler = Callable[[OutputMessage], Awaitable[None]]
ResultHandler = Callable[[SingleExecutionResult], Awaitable[None]]
CompleteHandler = Callable[[ExecutionComplete], Awaitable[None]]
ErrorHandler = Callable[[ExecutionError], Awaitable[None]]
InitHandler = Callable[[ExecutionInit], Awaitable[None]]


class ExecutionResultHandler(BaseModel):
    """
    Async callback container for sandbox execution events.

    This mirrors opensandbox's handler surface, but uses the project's own
    execution domain models so callers get typed payloads.
    """

    on_stdout: OutputHandler | None = Field(
        default=None, description="Async handler for stdout messages"
    )
    on_stderr: OutputHandler | None = Field(
        default=None, description="Async handler for stderr messages"
    )
    on_result: ResultHandler | None = Field(
        default=None, description="Async handler for execution results"
    )
    on_execution_complete: CompleteHandler | None = Field(
        default=None,
        description="Async handler for execution completion",
        alias="on_execution_complete",
    )
    on_error: ErrorHandler | None = Field(
        default=None, description="Async handler for execution errors"
    )
    on_init: InitHandler | None = Field(
        default=None, description="Async handler for execution init"
    )

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)