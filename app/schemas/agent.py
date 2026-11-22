from typing import Any

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    message: str = Field(
        min_length=3,
        max_length=2000,
        examples=["Compare how attention is used in Bahdanau et al. and in the Transformer."],
    )
    thread_id: str | None = Field(
        default=None, description="Continue a previous conversation (memory)."
    )


class ToolStepOut(BaseModel):
    tool: str
    args: dict[str, Any]
    result_preview: str


class AgentResponse(BaseModel):
    thread_id: str  # send it back to continue this conversation
    answer: str
    steps: list[ToolStepOut]  # every tool call the agent made, in order: the trace
    input_tokens: int
    output_tokens: int
    stopped_early: bool  # True if the step limit stopped it
