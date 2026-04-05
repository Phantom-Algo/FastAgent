from pydantic import BaseModel
from typing import List, Optional, TypedDict, Literal, Union, Any

class ResponseFormat(TypedDict):
    type: Literal["text", "json_object"]

class ExtraBodyThinking(TypedDict):
    type: Literal["disabled", "enabled", "auto"]

class ExtraBody(TypedDict):
    thinking: Union[ExtraBodyThinking, Any]

class BaseLLMConfig(BaseModel):
    model_name: str

    api_key: str

    base_url: str

    provider: str

    temperature: float = 0.7

    top_p: float = 0.9

    max_tokens: int = 1024

    frequency_penalty: float = 0.0

    presence_penalty: float = 0.0

    stop_sequences: List[str] = []

    tool_choice: str = "auto"

    parallel_tool_calls: bool = True

    response_format: Optional[ResponseFormat] = None

    extra_body: Optional[Union[ExtraBody, Any]] = None