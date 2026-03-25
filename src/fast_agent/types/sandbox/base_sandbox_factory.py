from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any

from .base_sandbox import ISandBox


class ISandBoxFactory(ABC):

    @abstractmethod
    async def create_sandbox(
        self,
        image: str,
        domain: str = "localhost:8080",
        api_key: str | None = None,
        *,
        protocol: str = "http",
        request_timeout: timedelta = timedelta(seconds=60),
        timeout: timedelta = timedelta(minutes=10),
        ready_timeout: timedelta = timedelta(seconds=30),
        env: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
        resource: dict[str, str] | None = None,
        network_policy: Any | None = None,
        extensions: dict[str, str] | None = None,
        entrypoint: list[str] | None = None,
        volumes: list[Any] | None = None,
        health_check: Any | None = None,
        health_check_polling_interval: timedelta = timedelta(milliseconds=200),
        skip_health_check: bool = False,
        debug: bool = False,
        user_agent: str = "FastAgent-OpenSandbox/1.0",
        headers: dict[str, str] | None = None,
        use_server_proxy: bool = False,
    ) -> ISandBox:
        """创建一个新的沙箱实例。"""

    @abstractmethod
    async def connect_sandbox(
        self,
        sandbox_id: str,
        domain: str = "localhost:8080",
        api_key: str | None = None,
        *,
        protocol: str = "http",
        request_timeout: timedelta = timedelta(seconds=60),
        connect_timeout: timedelta = timedelta(seconds=30),
        health_check_polling_interval: timedelta = timedelta(milliseconds=200),
        skip_health_check: bool = False,
        health_check: Any | None = None,
        debug: bool = False,
        user_agent: str = "FastAgent-OpenSandbox/1.0",
        headers: dict[str, str] | None = None,
        use_server_proxy: bool = False,
    ) -> ISandBox:
        """连接一个已存在的沙箱实例。"""

    @abstractmethod
    async def resume_sandbox(
        self,
        sandbox_id: str,
        domain: str = "localhost:8080",
        api_key: str | None = None,
        *,
        protocol: str = "http",
        request_timeout: timedelta = timedelta(seconds=60),
        resume_timeout: timedelta = timedelta(seconds=30),
        health_check_polling_interval: timedelta = timedelta(milliseconds=200),
        skip_health_check: bool = False,
        health_check: Any | None = None,
        debug: bool = False,
        user_agent: str = "FastAgent-OpenSandbox/1.0",
        headers: dict[str, str] | None = None,
        use_server_proxy: bool = False,
    ) -> ISandBox:
        """恢复一个已暂停的沙箱实例。"""