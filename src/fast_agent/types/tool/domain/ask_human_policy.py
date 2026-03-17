from pydantic import BaseModel

class AskHumanPolicy(BaseModel):
    """AskHumanPolicy 定义了当工具需要发起人工请求时的策略信息"""
    timeout: int

    