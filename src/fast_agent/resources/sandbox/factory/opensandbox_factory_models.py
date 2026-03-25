from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Optional

from opensandbox import Sandbox
from opensandbox.config import ConnectionConfig
from opensandbox.models.sandboxes import NetworkPolicy, Volume
from pydantic import BaseModel, ConfigDict, Field


class OpenSandboxConnectionOptions(BaseModel):
    """OpenSandbox 连接配置。"""

    domain: str = Field(default="localhost:8080", description="Sandbox API 域名")
    api_key: Optional[str] = Field(default=None, description="Sandbox API Key")
    protocol: str = Field(default="http", description="请求协议，支持 http/https")
    request_timeout: timedelta = Field(
        default=timedelta(seconds=60),
        description="管理 API 请求超时",
    )
    debug: bool = Field(default=False, description="是否开启 HTTP 调试日志")
    user_agent: str = Field(
        default="FastAgent-OpenSandbox/1.0",
        description="用户代理",
    )
    headers: dict[str, str] = Field(default_factory=dict, description="额外请求头")
    use_server_proxy: bool = Field(
        default=False,
        description="是否使用 server proxy 访问 execd",
    )

    @classmethod
    def from_env(cls) -> "OpenSandboxConnectionOptions":
        return cls(
            domain=os.getenv("SANDBOX_DOMAIN", "localhost:8080"),
            api_key=os.getenv("SANDBOX_API_KEY"),
        )

    def to_connection_config(self) -> ConnectionConfig:
        return ConnectionConfig(
            domain=self.domain,
            api_key=self.api_key,
            protocol=self.protocol,
            request_timeout=self.request_timeout,
            debug=self.debug,
            user_agent=self.user_agent,
            headers=self.headers,
            use_server_proxy=self.use_server_proxy,
        )


class OpenSandboxCreateOptions(BaseModel):
    """创建沙箱所需参数。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: str = Field(default="ubuntu:22.04", description="沙箱镜像")
    connection: OpenSandboxConnectionOptions = Field(
        default_factory=OpenSandboxConnectionOptions,
        description="连接配置",
    )
    timeout: timedelta = Field(default=timedelta(minutes=10), description="沙箱总生存时长")
    ready_timeout: timedelta = Field(
        default=timedelta(seconds=30),
        description="等待沙箱就绪超时",
    )
    env: dict[str, str] = Field(default_factory=dict, description="环境变量")
    metadata: dict[str, str] = Field(default_factory=dict, description="业务元数据")
    resource: dict[str, str] = Field(
        default_factory=lambda: {"cpu": "1", "memory": "2Gi"},
        description="资源限制",
    )
    network_policy: Optional[NetworkPolicy] = Field(default=None, description="网络策略")
    extensions: dict[str, str] = Field(default_factory=dict, description="扩展参数")
    entrypoint: Optional[list[str]] = Field(default=None, description="容器入口命令")
    volumes: Optional[list[Volume]] = Field(default=None, description="挂载卷")
    health_check: Optional[Callable[[Sandbox], Awaitable[bool]]] = Field(
        default=None,
        description="自定义健康检查函数",
    )
    health_check_polling_interval: timedelta = Field(
        default=timedelta(milliseconds=200),
        description="健康检查轮询间隔",
    )
    skip_health_check: bool = Field(default=False, description="是否跳过健康检查")

    @classmethod
    def from_env(cls) -> "OpenSandboxCreateOptions":
        return cls(
            image=os.getenv("SANDBOX_IMAGE", "ubuntu:22.04"),
            connection=OpenSandboxConnectionOptions.from_env(),
        )


class OpenSandboxConnectOptions(BaseModel):
    """连接已存在沙箱所需参数。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    connection: OpenSandboxConnectionOptions = Field(
        default_factory=OpenSandboxConnectionOptions,
        description="连接配置",
    )
    health_check: Optional[Callable[[Sandbox], Awaitable[bool]]] = Field(
        default=None,
        description="自定义健康检查函数",
    )
    connect_timeout: timedelta = Field(
        default=timedelta(seconds=30),
        description="连接后等待就绪超时",
    )
    health_check_polling_interval: timedelta = Field(
        default=timedelta(milliseconds=200),
        description="健康检查轮询间隔",
    )
    skip_health_check: bool = Field(default=False, description="是否跳过健康检查")


class OpenSandboxResumeOptions(BaseModel):
    """恢复已暂停沙箱所需参数。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    connection: OpenSandboxConnectionOptions = Field(
        default_factory=OpenSandboxConnectionOptions,
        description="连接配置",
    )
    health_check: Optional[Callable[[Sandbox], Awaitable[bool]]] = Field(
        default=None,
        description="自定义健康检查函数",
    )
    resume_timeout: timedelta = Field(
        default=timedelta(seconds=30),
        description="恢复后等待就绪超时",
    )
    health_check_polling_interval: timedelta = Field(
        default=timedelta(milliseconds=200),
        description="健康检查轮询间隔",
    )
    skip_health_check: bool = Field(default=False, description="是否跳过健康检查")
