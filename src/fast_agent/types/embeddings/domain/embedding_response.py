from typing import List, Literal, Optional, Union

from pydantic import BaseModel, model_validator


class EmbeddingVector(BaseModel):
    object: Literal["embedding"] = "embedding"

    index: int

    embedding: Union[List[float], str]


class EmbeddingUsage(BaseModel):
    prompt_tokens: int = 0

    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"

    data: List[EmbeddingVector]

    model: Optional[str] = None

    usage: Optional[EmbeddingUsage] = None

    @model_validator(mode="after")
    def check_data_not_empty(self):
        if not self.data:
            raise ValueError("EmbeddingResponse data must not be empty")
        return self