"""ORM models: Python classes mapped to PostgreSQL tables."""

import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Enum, Integer, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class DocumentStatus(enum.StrEnum):
    PENDING = "pending"  # saved, waiting for the background task
    PROCESSING = "processing"  # being extracted / chunked / embedded
    READY = "ready"  # vectors are in Qdrant, searchable
    FAILED = "failed"  # see the `error` column


class DocumentSource(enum.StrEnum):
    UPLOAD = "upload"
    ARXIV = "arxiv"


def _now() -> datetime:
    return datetime.now(UTC)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(500))
    source: Mapped[DocumentSource] = mapped_column(Enum(DocumentSource, native_enum=False))
    source_ref: Mapped[str] = mapped_column(String(500))  # file name or arXiv id
    # SHA-256 of the PDF bytes. UNIQUE = the database itself refuses duplicates.
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    file_path: Mapped[str] = mapped_column(String(1000))
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, native_enum=False), default=DocumentStatus.PENDING
    )
    error: Mapped[str | None] = mapped_column(Text, default=None)
    num_pages: Mapped[int | None] = mapped_column(Integer, default=None)
    num_chunks: Mapped[int | None] = mapped_column(Integer, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    ) 
