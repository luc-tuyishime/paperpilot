"""Repository: the only place that writes SQL for documents.

Routes and services call these methods instead of using the Session directly,
so database details stay in one file (same idea as a TypeORM repository in NestJS).
"""

import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Document, DocumentSource, DocumentStatus


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, document_id: uuid.UUID) -> Document | None:
        return self.session.get(Document, document_id)

    def get_by_hash(self, content_hash: str) -> Document | None:
        return self.session.scalar(select(Document).where(Document.content_hash == content_hash))

    def get_by_source(self, source: DocumentSource, source_ref: str) -> Document | None:
        return self.session.scalar(
            select(Document).where(Document.source == source, Document.source_ref == source_ref)
        )

    def rename(self, document: Document, title: str) -> None:
        document.title = title
        self.session.commit()

    def list(self) -> Sequence[Document]:
        return self.session.scalars(select(Document).order_by(Document.created_at.desc())).all()

    def add(self, document: Document) -> Document:
        self.session.add(document)
        self.session.commit()
        return document

    def set_status(
        self,
        document: Document,
        status: DocumentStatus,
        *,
        error: str | None = None,
        num_pages: int | None = None,
        num_chunks: int | None = None,
    ) -> None:
        document.status = status
        document.error = error
        if num_pages is not None:
            document.num_pages = num_pages
        if num_chunks is not None:
            document.num_chunks = num_chunks
        self.session.commit()
