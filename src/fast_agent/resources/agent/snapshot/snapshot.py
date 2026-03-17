"""
快照模块

Snapshot 是 BaseSnapshot 的具体实现，用于保存和恢复 Agent 的完整运行状态。
支持序列化/反序列化以便持久化存储或网络传输。
"""

from __future__ import annotations

from typing import Any
import pickle

from ....types.agent.snapshot.base_snapshot import BaseSnapshot


class Snapshot(BaseSnapshot):
    """
    快照具体实现，用于保存和恢复 Agent 的完整运行状态。

    包含字段：
    - llm_config: 大模型配置
    - context: 上下文（检查点时刻的干净副本）
    - lifespan_manager: 生命周期管理器
    - user_input: 当前轮次的用户输入
    - llm_output: 最近一次 LLM 输出
    - tool_results: 最近一次工具执行结果
    - tool_call_guard_triggered_contexts: Guard 触发的工具调用上下文列表
    - finished_tool_calls: Guard 过滤后已确认可执行的工具调用列表
    - state: 中断时所处的 FSM 状态阶段
    """

    def serialize(self) -> bytes:
        """
        将快照序列化为可持久化的字节格式。

        返回: 包含快照所有字段的字节数据
        """
        return pickle.dumps(self)

    @classmethod
    def deserialize(cls, data: Any) -> "Snapshot":
        """
        从字节数据反序列化为 Snapshot 实例。

        参数:
        - data: 包含快照字段的字节数据

        返回: Snapshot 实例
        """
        return pickle.loads(data)
