from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseAskHumanChannel(ABC):
    """
    BaseAskHumanChannel 定义了人类请求通道的抽象类
    """

    @abstractmethod
    async def ask_human(self, data: Dict[str, Any], timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        发起请求并等待人类响应
        
        @param data: 发送给人类的数据
        @param timeout: 等待响应的超时时间，单位为秒
        @return: 人类响应的数据
        """
        ...

