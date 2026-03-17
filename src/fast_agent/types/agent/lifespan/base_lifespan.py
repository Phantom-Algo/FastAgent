from abc import ABC, abstractmethod

from .dto.lifespan_dto import (
	AfterExecuteToolsRequest,
	AfterExecuteToolsResponse,
	AfterFinishRequest,
	AfterFinishResponse,
	AfterLLMOutputRequest,
	AfterLLMOutputResponse,
	AfterUserInputRequest,
	AfterUserInputResponse,
	BeforeExecuteToolsRequest,
	BeforeExecuteToolsResponse,
	ExecutingToolsRequest,
	ExecutingToolsResponse,
)


class IAfterFinish(ABC):
	@abstractmethod
	async def execute(self, data: AfterFinishRequest) -> AfterFinishResponse:
		...


class IAfterUserInput(ABC):
	@abstractmethod
	async def execute(self, data: AfterUserInputRequest) -> AfterUserInputResponse:
		...


class IAfterLLMOutput(ABC):
	@abstractmethod
	async def execute(self, data: AfterLLMOutputRequest) -> AfterLLMOutputResponse:
		...


class IBeforeExecuteTools(ABC):
	@abstractmethod
	async def execute(self, data: BeforeExecuteToolsRequest) -> BeforeExecuteToolsResponse:
		...


class IExecutingTools(ABC):
	@abstractmethod
	async def execute(self, data: ExecutingToolsRequest) -> ExecutingToolsResponse:
		...


class IAfterExecuteTools(ABC):
	@abstractmethod
	async def execute(self, data: AfterExecuteToolsRequest) -> AfterExecuteToolsResponse:
		...
