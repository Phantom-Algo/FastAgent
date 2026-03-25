from __future__ import annotations

from datetime import timedelta
from typing import Any, Optional

from opensandbox import Sandbox
from opensandbox.models.execd import (
	Execution as OSExecution,
	ExecutionComplete as OSExecutionComplete,
	ExecutionError as OSExecutionError,
	ExecutionHandlers as OSExecutionHandlers,
	ExecutionInit as OSExecutionInit,
	ExecutionResult as OSExecutionResult,
	OutputMessage as OSOutputMessage,
	RunCommandOpts as OSRunCommandOpts,
)

from ....types.sandbox.base_sandbox import ISandBox
from ....types.sandbox.domain.command_options import CommandOpts
from ....types.sandbox.domain.execution_result import (
	ExecutionComplete,
	ExecutionError,
	ExecutionInit,
	ExecutionResult,
	OutputMessage,
	SingleExecutionResult,
)
from ....types.sandbox.domain.execution_result_handler import ExecutionResultHandler


class OpenSandboxInstance(ISandBox):
	"""OpenSandbox 的 ISandBox 适配实现。"""

	def __init__(self, sandbox: Sandbox) -> None:
		self._sandbox = sandbox

	@property
	def id(self) -> str:
		return self._sandbox.id

	async def command(
		self,
		cmd: str,
		*,
		opts: Optional[CommandOpts] = None,
		execution_result_handler: Optional[ExecutionResultHandler] = None,
	) -> ExecutionResult:
		os_opts = self._to_os_command_opts(opts)
		os_handlers = self._to_os_execution_handlers(execution_result_handler)
		execution = await self._sandbox.commands.run(
			cmd,
			opts=os_opts,
			handlers=os_handlers,
		)
		return self._from_os_execution(execution)

	async def pause(self) -> None:
		await self._sandbox.pause()

	async def kill(self) -> None:
		await self._sandbox.kill()

	async def close(self) -> None:
		await self._sandbox.close()

	async def renew(self, timeout: timedelta) -> Any:
		return await self._sandbox.renew(timeout)

	async def is_healthy(self) -> bool:
		return await self._sandbox.is_healthy()

	async def check_ready(
		self,
		timeout: timedelta,
		polling_interval: timedelta,
	) -> None:
		await self._sandbox.check_ready(timeout, polling_interval)


    # ===== OpenSandbox <-> ISandBox 适配器的内部转换方法 =====
	def _to_os_command_opts(self, opts: Optional[CommandOpts]) -> Optional[OSRunCommandOpts]:
		if opts is None:
			return None

		return OSRunCommandOpts(
			background=opts.background,
			working_directory=opts.working_directory,
			timeout=opts.timeout,
		)

	def _to_os_execution_handlers(
		self,
		handler: Optional[ExecutionResultHandler],
	) -> Optional[OSExecutionHandlers]:
		if handler is None:
			return None

		async def on_stdout(message: OSOutputMessage) -> None:
			if handler.on_stdout is None:
				return
			await handler.on_stdout(self._from_os_output_message(message))

		async def on_stderr(message: OSOutputMessage) -> None:
			if handler.on_stderr is None:
				return
			await handler.on_stderr(self._from_os_output_message(message))

		async def on_result(result: OSExecutionResult) -> None:
			if handler.on_result is None:
				return
			await handler.on_result(self._from_os_single_result(result))

		async def on_execution_complete(event: OSExecutionComplete) -> None:
			if handler.on_execution_complete is None:
				return
			await handler.on_execution_complete(
				ExecutionComplete(
					timestamp=event.timestamp,
					execution_time_in_millis=event.execution_time_in_millis,
				)
			)

		async def on_error(error: OSExecutionError) -> None:
			if handler.on_error is None:
				return
			await handler.on_error(
				ExecutionError(
					name=error.name,
					value=error.value,
					timestamp=error.timestamp,
					traceback=list(error.traceback),
				)
			)

		async def on_init(event: OSExecutionInit) -> None:
			if handler.on_init is None:
				return
			await handler.on_init(ExecutionInit(id=event.id, timestamp=event.timestamp))

		return OSExecutionHandlers(
			on_stdout=on_stdout if handler.on_stdout else None,
			on_stderr=on_stderr if handler.on_stderr else None,
			on_result=on_result if handler.on_result else None,
			on_execution_complete=(
				on_execution_complete if handler.on_execution_complete else None
			),
			on_error=on_error if handler.on_error else None,
			on_init=on_init if handler.on_init else None,
		)

	def _from_os_execution(self, execution: OSExecution) -> ExecutionResult:
		result = ExecutionResult(
			id=execution.id,
			execution_count=execution.execution_count,
		)

		for item in execution.result:
			result.add_result(self._from_os_single_result(item))

		if execution.error is not None:
			result.error = ExecutionError(
				name=execution.error.name,
				value=execution.error.value,
				timestamp=execution.error.timestamp,
				traceback=list(execution.error.traceback),
			)

		for stdout_message in execution.logs.stdout:
			result.logs.add_stdout(self._from_os_output_message(stdout_message))

		for stderr_message in execution.logs.stderr:
			result.logs.add_stderr(self._from_os_output_message(stderr_message))

		return result

	def _from_os_single_result(self, item: OSExecutionResult) -> SingleExecutionResult:
		return SingleExecutionResult(
			text=item.text,
			timestamp=item.timestamp,
			extra_properties=dict(item.extra_properties),
		)

	def _from_os_output_message(self, message: OSOutputMessage) -> OutputMessage:
		return OutputMessage(
			text=message.text,
			timestamp=message.timestamp,
			is_error=message.is_error,
		)
