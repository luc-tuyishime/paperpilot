"""The ingestion pipeline. Runs in the background after the upload request has returned.

    PDF on disk -> pages -> chunks -> embeddings (batched) -> Qdrant, status in PostgreSQL

Important: the background task opens its OWN database session. The request's session
is already closed by the time this runs.
"""

import logging
import uuid
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.db.models import DocumentStatus
from app.repositories.documents import DocumentRepository
from app.services.chunking import chunk_pages
from app.services.embeddings import Embedder
from app.services.pdf import extract_pages
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


def ingest_document(
    document_id: uuid.UUID,
    session_factory: sessionmaker[Session],
    embedder: Embedder,
    store: VectorStore,
    settings: Settings,
) -> None:
    with session_factory() as session:
        repo = DocumentRepository(session)
        document = repo.get(document_id)
        if document is None:
            logger.warning("document %s disappeared before ingestion", document_id)
            return

        repo.set_status(document, DocumentStatus.PROCESSING)
        try:
            pages = extract_pages(Path(document.file_path))
            if not pages:
                raise ValueError("no extractable text (is it a scanned PDF?)")
            chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)

            vectors: list[list[float]] = []
            batch = settings.embed_batch_size
            for i in range(0, len(chunks), batch):
                vectors.extend(embedder.embed_documents([c.text for c in chunks[i : i + batch]]))

            store.ensure_collection(dimension=len(vectors[0]))
            store.delete_document(document.id)  # clean slate: re-ingestion never duplicates
            store.upsert_chunks(document.id, document.title, chunks, vectors)

            repo.set_status(
                document, DocumentStatus.READY, num_pages=len(pages), num_chunks=len(chunks)
            )
            logger.info("ingested %s: %d pages, %d chunks", document.title, len(pages), len(chunks))
        except Exception as exc:  # any failure -> visible status, never a silent crash
            logger.exception("ingestion failed for %s", document_id)
            repo.set_status(document, DocumentStatus.FAILED, error=str(exc)[:1000])
