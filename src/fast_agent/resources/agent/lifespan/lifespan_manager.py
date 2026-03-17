"""
生命周期管理器模块

LifespanManager 是 BaseLifespanManager 的具体实现，负责：
- 管理 6 个生命周期 handler 的注册、获取、移除
- 管理跨生命周期阶段共享的 kwargs 字典
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Union

from ....types.agent.lifespan.base_lifespan import (
    IAfterExecuteTools,
    IAfterFinish,
    IAfterLLMOutput,
    IAfterUserInput,
    IBeforeExecuteTools,
    IExecutingTools,
)
from ....types.agent.lifespan.base_lifespan_manager import BaseLifespanManager
from ....types.agent.lifespan.enum.lifespan_type_enum import LifespanType


# 生命周期类型字面量别名，与枚举保持一致
LifespanTypeLiteral = Literal[
    "after_finish",
    "after_user_input",
    "after_llm_output",
    "before_execute_tools",
    "executing_tools",
    "after_execute_tools",
]

# LifespanType 枚举值 → 属性名的映射
_TYPE_TO_ATTR: Dict[LifespanType, str] = {
    LifespanType.AFTER_FINISH: "after_finish",
    LifespanType.AFTER_USER_INPUT: "after_user_input",
    LifespanType.AFTER_LLM_OUTPUT: "after_llm_output",
    LifespanType.BEFORE_EXECUTE_TOOLS: "before_execute_tools",
    LifespanType.EXECUTING_TOOLS: "executing_tools",
    LifespanType.AFTER_EXECUTE_TOOLS: "after_execute_tools",
}


class LifespanManager(BaseLifespanManager):
    """
    生命周期管理器，管理 Agent 运行各阶段的 handler 注册与调度。

    支持 6 个生命周期阶段的 handler 动态注册和替换：
    - after_user_input: 用户输入后
    - after_llm_output: LLM 输出后
    - before_execute_tools: 工具执行前（Guard 检测）
    - executing_tools: 工具执行中
    - after_execute_tools: 工具执行后
    - after_finish: 轮次结束后
    """

    def __init__(
        self,
        after_finish: Optional[IAfterFinish] = None,
        after_user_input: Optional[IAfterUserInput] = None,
        after_llm_output: Optional[IAfterLLMOutput] = None,
        before_execute_tools: Optional[IBeforeExecuteTools] = None,
        executing_tools: Optional[IExecutingTools] = None,
        after_execute_tools: Optional[IAfterExecuteTools] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        # 延迟导入默认实现，避免循环导入
        from .default_lifespan import (
            DefaultAfterExecuteTools,
            DefaultAfterFinish,
            DefaultAfterLLMOutput,
            DefaultAfterUserInput,
            DefaultBeforeExecuteTools,
            DefaultExecutingTools,
        )

        self.after_finish: IAfterFinish = after_finish or DefaultAfterFinish()
        self.after_user_input: IAfterUserInput = after_user_input or DefaultAfterUserInput()
        self.after_llm_output: IAfterLLMOutput = after_llm_output or DefaultAfterLLMOutput()
        self.before_execute_tools: IBeforeExecuteTools = before_execute_tools or DefaultBeforeExecuteTools()
        self.executing_tools: IExecutingTools = executing_tools or DefaultExecutingTools()
        self.after_execute_tools: IAfterExecuteTools = after_execute_tools or DefaultAfterExecuteTools()
        self.kwargs: Dict[str, Any] = kwargs if kwargs is not None else {}

    # ===== kwargs 管理 =====

    def get_kwargs(self) -> Dict[str, Any]:
        return self.kwargs

    def set_kwargs(self, kwargs: Dict[str, Any]) -> None:
        self.kwargs = kwargs

    def update_kwargs(self, kwargs: Dict[str, Any]) -> None:
        self.kwargs.update(kwargs)

    # ===== 生命周期 handler 管理 =====

    def _normalize_type(self, lifespan_type: Union[LifespanType, LifespanTypeLiteral]) -> LifespanType:
        """将字符串字面量统一转换为 LifespanType 枚举。"""
        if isinstance(lifespan_type, str):
            return LifespanType(lifespan_type)
        return lifespan_type

    def set_lifespan(
        self,
        lifespan_type: Union[LifespanType, LifespanTypeLiteral],
        handler: Any,
    ) -> None:
        """注册指定阶段的生命周期 handler。"""
        enum_type = self._normalize_type(lifespan_type)
        attr_name = _TYPE_TO_ATTR.get(enum_type)
        if attr_name is None:
            raise ValueError(f"Unsupported lifespan type: {lifespan_type}")
        setattr(self, attr_name, handler)

    def get_lifespan(
        self,
        lifespan_type: Union[LifespanType, LifespanTypeLiteral],
    ) -> Optional[Any]:
        """获取指定阶段的生命周期 handler，未注册则返回 None。"""
        enum_type = self._normalize_type(lifespan_type)
        attr_name = _TYPE_TO_ATTR.get(enum_type)
        if attr_name is None:
            raise ValueError(f"Unsupported lifespan type: {lifespan_type}")
        return getattr(self, attr_name, None)

    def remove_lifespan(
        self,
        lifespan_type: Union[LifespanType, LifespanTypeLiteral],
    ) -> None:
        """移除指定阶段的生命周期 handler，恢复为默认实现。"""
        from .default_lifespan import (
            DefaultAfterExecuteTools,
            DefaultAfterFinish,
            DefaultAfterLLMOutput,
            DefaultAfterUserInput,
            DefaultBeforeExecuteTools,
            DefaultExecutingTools,
        )

        # 默认实现映射
        _defaults: Dict[LifespanType, Any] = {
            LifespanType.AFTER_FINISH: DefaultAfterFinish,
            LifespanType.AFTER_USER_INPUT: DefaultAfterUserInput,
            LifespanType.AFTER_LLM_OUTPUT: DefaultAfterLLMOutput,
            LifespanType.BEFORE_EXECUTE_TOOLS: DefaultBeforeExecuteTools,
            LifespanType.EXECUTING_TOOLS: DefaultExecutingTools,
            LifespanType.AFTER_EXECUTE_TOOLS: DefaultAfterExecuteTools,
        }

        enum_type = self._normalize_type(lifespan_type)
        attr_name = _TYPE_TO_ATTR.get(enum_type)
        if attr_name is None:
            raise ValueError(f"Unsupported lifespan type: {lifespan_type}")

        default_cls = _defaults[enum_type]
        setattr(self, attr_name, default_cls())
