import uuid

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    # Guardrail: bounded input. Long inputs cost tokens and are a common injection vector.
    question: str = Field(
        min_length=3, max_length=1000, examples=["What BLEU score does the Transformer get?"]
    )
    top_k: int | None = Field(default=None, ge=1, le=12)


class Source(BaseModel):
    number: int  # the [n] used in the answer
    document_id: uuid.UUID
    title: str
    page: int
    score: float
    text: str


class UsageOut(BaseModel):
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int


class AskResponse(BaseModel):
    answer: str
    grounded: bool  # False when we refused: nothing relevant was retrieved, or the model said IDK
    citations: list[Source]  # sources the answer actually cites
    sources: list[Source]  # everything that was given to the model
    best_score: float
    usage: UsageOut | None  # None when the LLM wasn't called
