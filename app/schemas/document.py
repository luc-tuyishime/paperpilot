"""API schemas (what clients send and receive). Separate from the ORM models on purpose:
the database can change without breaking the API contract."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import DocumentSource, DocumentStatus


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # build it from an ORM object

    id: uuid.UUID
    title: str
    source: DocumentSource
    source_ref: str
    status: DocumentStatus
    error: str | None
    num_pages: int | None
    num_chunks: int | None
    created_at: datetime


class ArxivImport(BaseModel):
    # e.g. "1706.03762" or "1706.03762v7"
    arxiv_id: str = Field(pattern=r"^\d{4}\.\d{4,5}(v\d+)?$", examples=["1706.03762"])


class DocumentUpdate(BaseModel):
    title: str = Field(min_length=3, max_length=500)


class SearchHit(BaseModel):
    document_id: uuid.UUID
    title: str
    page: int
    chunk_index: int
    score: float
    text: str
