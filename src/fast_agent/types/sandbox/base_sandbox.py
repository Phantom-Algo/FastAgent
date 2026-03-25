from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, Optional

from .domain.command_options import CommandOpts
from .domain.execution_result import ExecutionResult
from .domain.execution_result_handler import ExecutionResultHandler


class ISandBox(ABC):

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique sandbox identifier."""

    @abstractmethod
    async def command(
        self,
        cmd: str,
        *,
        opts: Optional[CommandOpts] = None,
        execution_result_handler: Optional[ExecutionResultHandler] = None,
    ) -> ExecutionResult:
        """
        执行 SHELL 命令并返回结果。
        """

    @abstractmethod
    async def pause(self) -> None:
        """暂停沙箱，同时保留其状态。"""

    @abstractmethod
    async def kill(self) -> None:
        """终止沙箱实例。"""

    @abstractmethod
    async def close(self) -> None:
        """释放与此沙箱客户端关联的本地资源。"""

    @abstractmethod
    async def renew(self, timeout: timedelta) -> Any:
        """
        延长沙箱的过期时间。

        返回类型故意保持开放，因为不同的提供者可能返回不同的续订负载模型。
        """

    @abstractmethod
    async def is_healthy(self) -> bool:
        """检查沙箱是否健康且响应正常。"""

    @abstractmethod
    async def check_ready(
        self,
        timeout: timedelta,
        polling_interval: timedelta,
    ) -> None:
        """等待沙箱在超时时间内变为就绪状态。"""

    async def __aenter__(self) -> "ISandBox":
        """允许使用 `async with` 进行生命周期安全的沙箱处理。"""
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        """在上下文退出时始终释放本地客户端资源。"""
        await self.close()