from typing import Any, Optional, Callable
from pydantic import BaseModel, ConfigDict, Field
from ...messages.domain.tool_result_message import ToolResultMessage

class GuardPolicyHumanResponseSchema(BaseModel):
    """GuardPolicyHumanResponseSchema 定义了当工具调用护栏触发后，人工干预的响应参数类型"""
    ...

class GuardPolicy(BaseModel):
    """
    GuardPolicy 定义了工具调用护栏的策略信息
    
    @param info: 护栏的描述信息
    @param response_schema: 护栏触发后人工干预的响应参数类型
    @param guard_func: 可选的护栏函数，用于判断人工干预的响应是否满足条件，返回 True 则认为护栏通过，返回 False 则认为护栏未通过
    """
    model_config = ConfigDict(populate_by_name=True)

    info: Optional[Any] = None

    response_schema: GuardPolicyHumanResponseSchema = Field(alias="schema")

    guard_func: Optional[Callable[[GuardPolicyHumanResponseSchema], bool]] = None

    reject_func: Optional[Callable[[GuardPolicyHumanResponseSchema], ToolResultMessage]] = None

    @property
    def schema(self) -> GuardPolicyHumanResponseSchema:
        return self.response_schema