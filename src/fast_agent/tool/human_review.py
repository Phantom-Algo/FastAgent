from fast_agent.agent.event import EventChannel, HumanReviewEvent, HumanResponseEvent
from typing import Dict, Any, Optional
import asyncio

class HumanReviewChannel:
    """
    人工审查通道类
    """
    def __init__(self, event_channel: EventChannel):
        self.event_channel = event_channel
        self.response_channel: Optional[asyncio.Future] = None  # 用于接收人工审查响应的通道

    async def request(self, data: Dict[str, Any], timeout: Optional[int] = None) -> Dict[str, Any]:
        """ 发送人工审查请求并等待响应 """
        # 创建一个新的 Future 来接收响应，确保每次请求都是独立的
        self.response_channel = asyncio.get_event_loop().create_future()

        event = HumanReviewEvent(
            data=HumanReviewEvent.HumanReviewEventData(
                content=data,
                response_channel=self.response_channel
            )
        )
        try:
            await self.event_channel.send_event(event)
            response: HumanResponseEvent = await asyncio.wait_for(self.response_channel, timeout=timeout)
            self._validate_response(response)
            return response.data.content
        except Exception as e:
            # 抛出的异常可能是因为超时 / 通道关闭 / 人工审查错误
            raise e
        finally:
            if self.response_channel and not self.response_channel.done():
                self.response_channel.set_result(None)  # 确保 Future 被标记为完成
            self.response_channel = None  # 清理响应通道
    
    def _validate_response(self, response: HumanResponseEvent):
        """ 校验人工审查响应 """
        if not isinstance(response, HumanResponseEvent):
            raise ValueError("Invalid response type, expected HumanResponseEvent.")
        if response.type == "human_normal_response":
            return
        elif response.type == "human_error_response":
            raise ValueError(f"Human review error: {response.data.message}")    




    
