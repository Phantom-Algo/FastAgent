from enum import Enum

class AgentRoundStopEnum(str, Enum):
    """Agent 单轮交互结束原因枚举"""
    UNKNOWN = "unknown"
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    BALANCE = "balance"
    ERROR = "error"