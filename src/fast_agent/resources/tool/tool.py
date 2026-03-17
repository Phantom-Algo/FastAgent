from ...types.tool.base_tool import BaseTool
from typing import Dict, Any


class Tool(BaseTool):

    def _build_parameters_schema(self) -> Dict[str, Any]:
        schema = self.args_schema.model_json_schema()
        schema.pop("title", None)
        return schema

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._build_parameters_schema(),
            },
        }

    def to_anthropic_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self._build_parameters_schema(),
        }

    def to_google_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._build_parameters_schema(),
        }

    