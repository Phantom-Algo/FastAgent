from typing import Type, List, Dict, Any, Callable, Optional
from pydantic import BaseModel, Field
from .domain.ask_human_policy import AskHumanPolicy
from .domain.guard_policy import GuardPolicy
import uuid

class BaseTool(BaseModel):
    """BaseTool 标准化工具定义抽象类"""
    id: str = Field(default_factory=lambda: f"tool_{uuid.uuid4().hex[:16]}")

    name: str

    description: str

    args_schema: Type[BaseModel]

    func: Callable[..., Any]

    is_async: bool

    labels: Optional[List[str]] = None

    inject_params: Optional[List[str]] = None

    tool_runtime_param_name: Optional[str] = None

    ask_human_policy: Optional[AskHumanPolicy] = None

    guard_policy: Optional[GuardPolicy] = None

    def __call__(self, *args, **kwds):
        return self.func(*args, **kwds)
    
    def __repr__(self):
        return f"<Tool: {self.name}, id={self.id}>"

    def to_openai_schema(self) -> Dict[str, Any]:
        ...

    def to_anthropic_schema(self) -> Dict[str, Any]:
        ...

    def to_google_schema(self) -> Dict[str, Any]:
        ...

