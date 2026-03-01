import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fast_agent.tool import BaseTool, tool


def test_tool_decorator_builds_args_schema_and_filters_inject_params() -> None:
    @tool(tool_name="add_numbers", inject_params=["runtime_ctx"], labels=["math"])
    def add(a: int, b: int, runtime_ctx: dict) -> int:
        """两个整数相加。"""
        return a + b

    assert isinstance(add, BaseTool)
    assert add.name == "add_numbers"
    assert add.description == "两个整数相加。"
    assert add.labels == ["math"]
    assert set(add.inject_params or []) == {"runtime_ctx"}
    assert add.is_async is False

    args_schema = add.args_schema.model_json_schema()
    assert set(args_schema["properties"].keys()) == {"a", "b"}
    assert set(args_schema.get("required", [])) == {"a", "b"}
    assert "runtime_ctx" not in args_schema["properties"]

    assert add(a=1, b=2, runtime_ctx={"trace_id": "t1"}) == 3


def test_async_tool_decorator_works() -> None:
    @tool(inject_params=["runtime_ctx"])
    async def greet(name: str, runtime_ctx: dict) -> str:
        """异步打招呼。"""
        return f"hello {name}"

    assert greet.is_async is True
    assert set(greet.args_schema.model_json_schema()["properties"].keys()) == {"name"}

    result = asyncio.run(greet(name="fast-agent", runtime_ctx={"request_id": "r1"}))
    assert result == "hello fast-agent"


def test_tool_schema_conversion_for_three_vendors() -> None:
    @tool(tool_name="search_docs")
    def search(query: str, top_k: int = 5) -> str:
        """搜索文档。"""
        return f"query={query}, top_k={top_k}"

    openai_schema = search.to_openai_schema()
    assert openai_schema["type"] == "function"
    assert openai_schema["function"]["name"] == "search_docs"
    assert openai_schema["function"]["parameters"]["type"] == "object"
    assert "title" not in openai_schema["function"]["parameters"]

    anthropic_schema = search.to_anthropic_schema()
    assert anthropic_schema["name"] == "search_docs"
    assert anthropic_schema["input_schema"]["type"] == "object"

    google_schema = search.to_google_schema()
    assert google_schema["name"] == "search_docs"
    assert google_schema["parameters"]["type"] == "object"


def test_inject_params_unknown_key_raises_error() -> None:
    with pytest.raises(ValueError):
        @tool(inject_params=["missing"])
        def invalid(x: int) -> int:
            return x
