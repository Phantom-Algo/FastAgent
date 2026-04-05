from typing import Any, Literal, Optional

from pydantic import BaseModel


class BaseEmbeddingConfig(BaseModel):
    model_name: str

    api_key: str

    base_url: str

    provider: str

    dimensions: Optional[int] = None

    encoding_format: Literal["float", "base64"] = "float"

    user: Optional[str] = None

    extra_body: Optional[Any] = None