from pydantic import BaseModel, Field
from typing import Optional
from datetime import timedelta

class CommandOpts(BaseModel):

    background: bool = Field(
        default=False,
        description="Whether to run the command in the background (detached mode). ",
    )

    working_directory: Optional[str] = Field(
        default=None,
        description="The working directory to execute the command in."
    )

    timeout: Optional[timedelta] = Field(
        default=None,
        description="Maximum execution time; the server will terminate the command when reached. If omitted, the server will not enforce any timeout.",
    )