from ....types.agent.event.base_event_channel import BaseEventChannel
from ....types.agent.event.base_event import BaseEvent
from ....types.agent.exceptions.event_exception import EventChannelClosedException
from typing import Optional
import asyncio

class EventChannel(BaseEventChannel):

    def __init__(self):
        self._event_queue = asyncio.Queue()
        self._close_sentinel = object()
        self._closed = False

    async def send_event(self, event: BaseEvent) -> None:
        """发送事件到通道"""
        if self._closed:
            raise EventChannelClosedException("EventChannel is closed, cannot send event.")
        await self._event_queue.put(event)

    async def receive_event(self, timeout: Optional[int] = None) -> BaseEvent:
        """
        从通道接收事件（阻塞）
        
        参数说明：
        - timeout: 接收事件的超时时间（秒），默认为 None 表示无限等待
        """
        try:
            event = await asyncio.wait_for(self._event_queue.get(), timeout)
        except asyncio.TimeoutError:
            raise EventChannelClosedException("EventChannel receive_event timeout.")

        if event is self._close_sentinel:
            self._event_queue.task_done()
            raise EventChannelClosedException("EventChannel is closed.")

        return event

    def close(self) -> None:
        """关闭事件通道（幂等）。"""
        if self._closed:
            return
        self._closed = True
        self._event_queue.put_nowait(self._close_sentinel)

    @property
    def is_closed(self) -> bool:
        """事件通道是否已关闭。"""
        return self._closed
    
    def task_done(self):
        """标记事件处理完成"""
        self._event_queue.task_done()