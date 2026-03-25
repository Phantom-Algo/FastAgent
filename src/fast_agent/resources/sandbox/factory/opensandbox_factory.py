from __future__ import annotations

from datetime import timedelta
from typing import Any

from opensandbox import Sandbox
from opensandbox.models.sandboxes import NetworkPolicy, Volume

from ....types.sandbox.base_sandbox import ISandBox
from ....types.sandbox.base_sandbox_factory import ISandBoxFactory
from ..instance.opensandbox import OpenSandboxInstance
from .opensandbox_factory_models import (
	OpenSandboxConnectionOptions,
	OpenSandboxConnectOptions,
	OpenSandboxCreateOptions,
	OpenSandboxResumeOptions,
)


class OpenSandboxFactory(ISandBoxFactory):
	"""OpenSandbox 的沙箱工厂实现。"""

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
		network_policy: NetworkPolicy | None = None,
		extensions: dict[str, str] | None = None,
		entrypoint: list[str] | None = None,
		volumes: list[Volume] | None = None,
		health_check: Any | None = None,
		health_check_polling_interval: timedelta = timedelta(milliseconds=200),
		skip_health_check: bool = False,
		debug: bool = False,
		user_agent: str = "FastAgent-OpenSandbox/1.0",
		headers: dict[str, str] | None = None,
		use_server_proxy: bool = False,
	) -> ISandBox:
		create_config = OpenSandboxCreateOptions(
			image=image,
			connection=OpenSandboxConnectionOptions(
				domain=domain,
				api_key=api_key,
				protocol=protocol,
				request_timeout=request_timeout,
				debug=debug,
				user_agent=user_agent,
				headers=headers or {},
				use_server_proxy=use_server_proxy,
			),
			timeout=timeout,
			ready_timeout=ready_timeout,
			env=env or {},
			metadata=metadata or {},
			resource=resource or {"cpu": "1", "memory": "2Gi"},
			network_policy=network_policy,
			extensions=extensions or {},
			entrypoint=entrypoint,
			volumes=volumes,
			health_check=health_check,
			health_check_polling_interval=health_check_polling_interval,
			skip_health_check=skip_health_check,
		)

		sandbox = await Sandbox.create(
			create_config.image,
			timeout=create_config.timeout,
			ready_timeout=create_config.ready_timeout,
			env=create_config.env,
			metadata=create_config.metadata,
			resource=create_config.resource,
			network_policy=create_config.network_policy,
			extensions=create_config.extensions,
			entrypoint=create_config.entrypoint,
			volumes=create_config.volumes,
			connection_config=create_config.connection.to_connection_config(),
			health_check=create_config.health_check,
			health_check_polling_interval=create_config.health_check_polling_interval,
			skip_health_check=create_config.skip_health_check,
		)
		return OpenSandboxInstance(sandbox)

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
		config = OpenSandboxConnectOptions(
			connection=OpenSandboxConnectionOptions(
				domain=domain,
				api_key=api_key,
				protocol=protocol,
				request_timeout=request_timeout,
				debug=debug,
				user_agent=user_agent,
				headers=headers or {},
				use_server_proxy=use_server_proxy,
			),
			health_check=health_check,
			connect_timeout=connect_timeout,
			health_check_polling_interval=health_check_polling_interval,
			skip_health_check=skip_health_check,
		)

		sandbox = await Sandbox.connect(
			sandbox_id,
			connection_config=config.connection.to_connection_config(),
			health_check=config.health_check,
			connect_timeout=config.connect_timeout,
			health_check_polling_interval=config.health_check_polling_interval,
			skip_health_check=config.skip_health_check,
		)
		return OpenSandboxInstance(sandbox)

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
		config = OpenSandboxResumeOptions(
			connection=OpenSandboxConnectionOptions(
				domain=domain,
				api_key=api_key,
				protocol=protocol,
				request_timeout=request_timeout,
				debug=debug,
				user_agent=user_agent,
				headers=headers or {},
				use_server_proxy=use_server_proxy,
			),
			health_check=health_check,
			resume_timeout=resume_timeout,
			health_check_polling_interval=health_check_polling_interval,
			skip_health_check=skip_health_check,
		)

		sandbox = await Sandbox.resume(
			sandbox_id,
			connection_config=config.connection.to_connection_config(),
			health_check=config.health_check,
			resume_timeout=config.resume_timeout,
			health_check_polling_interval=config.health_check_polling_interval,
			skip_health_check=config.skip_health_check,
		)
		return OpenSandboxInstance(sandbox)
