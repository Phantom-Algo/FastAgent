from ...types.tool.base_ask_human_channel import BaseAskHumanChannel
from ...types.agent.event.base_event_channel import BaseEventChannel
from ...types.constant import DEFAULT_ASK_HUMAN_TIMEOUT
from ..agent.event.events import AskHumanEvent, AskHumanResponseEvent
from typing import Dict, Any, Optional
import asyncio

class AskHumanChannel(BaseAskHumanChannel):

    def __init__(
        self, 
        event_channel: BaseEventChannel
    ):
        self.event_channel = event_channel
        self.ask_human_response_channel: Optional[asyncio.Future] = None

    async def ask_human(self, data: Dict[str, Any], timeout: Optional[int] = None) -> Dict[str, Any]:
        if timeout is None:
            timeout = DEFAULT_ASK_HUMAN_TIMEOUT  # 默认超时时间为 300 秒

        self.ask_human_response_channel = asyncio.get_event_loop().create_future()

        ask_human_event = AskHumanEvent(
            data=AskHumanEvent.AskHumanEventData(
                content=data,
                ask_human_response_channel=self.ask_human_response_channel
            )
        )

        try:
            await self.event_channel.send_event(ask_human_event)
            response: AskHumanResponseEvent = await asyncio.wait_for(self.ask_human_response_channel, timeout=timeout)
            self._validate_response(response)
            return response.data.response_content

        except Exception as e:
            raise e
        
        finally:
            if self.ask_human_response_channel and not self.ask_human_response_channel.done():
                self.ask_human_response_channel.set_result(None)
            self.ask_human_response_channel = None

    def _validate_response(self, response: AskHumanResponseEvent):
        if not isinstance(response, AskHumanResponseEvent):
            raise ValueError("Invalid response type, expected AskHumanResponseEvent")
        
        if not response.data.response_success:
            raise ValueError(f"Human response indicates failure: {response.data.message}")
        
        return



        