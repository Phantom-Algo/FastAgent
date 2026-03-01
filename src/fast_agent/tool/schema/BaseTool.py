from pydantic import BaseModel, Field, ConfigDict
from typing import Callable, Any, List, Optional, Dict, Type, Union
import uuid

class BaseTool(BaseModel):
    """BaseTool 工具标准化 schema。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: f"tool_{str(uuid.uuid4().hex[:16])}")

    name: str
    description: str
    args_schema: Type[BaseModel]

    func: Callable[..., Any]
    is_async: bool

    labels: Optional[List[str]] = None

    inject_params: Optional[List[str]] = None

    def __call__(self, *args, **kwds):
        return self.func(*args, **kwds)
    
    def __repr__(self):
        return f"<Tool: f{self.name}, id={self.id}>"
    
    def _get_json_schema(self, *, clean: bool = True) -> Dict[str, Any]:
        """工具函数：将 args_schema 转换为 JSON Schema 格式。"""
        schema = self.args_schema.model_json_schema()
        if clean:
            schema = _clean_json_schema(schema)

        json_schema = {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", [])
        }

        if "$defs" in schema:
            json_schema["$defs"] = schema["$defs"]

        return json_schema

    def to_openai_schema(self) -> Dict[str, Any]:
        """转换为 OpenAI Responses/Chat Completions 的工具 schema 格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._get_json_schema()
            }
        }

    def to_anthropic_schema(self) -> Dict[str, Any]:
        """转换为 Anthropic Messages API 的工具 schema 格式。"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self._get_json_schema()
        }

    def to_google_schema(self) -> Dict[str, Any]:
        """转换为 Google/Gemini Function Calling 的工具 schema 格式。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_json_schema()
        }


def _clean_json_schema(schema: Union[Dict[str, Any], List[Any]]) -> Union[Dict[str, Any], List[Any]]:
    """递归清洗 JSON Schema，移除无关的 `title` 字段，避免提示词噪音。"""
    if isinstance(schema, dict):
        schema.pop("title", None)
        for value in schema.values():
            _clean_json_schema(value)
        return schema

    if isinstance(schema, list):
        for item in schema:
            _clean_json_schema(item)

    return schema

    