"""Document endpoints: upload a PDF, import from arXiv, list, get, rename, re-ingest.

Every new document is saved as PENDING and the response returns immediately (201).
The slow part (extract, embed, index) runs afterwards as a background task.
Clients poll GET /documents/{id} to watch the status move to READY.
"""

import hashlib
import uuid
from typing import Annotated

import httpx
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Form,
    HTTPException,
    Response,
    UploadFile,
    status,
)

from app.api.deps import EmbedderDep, RepoDep, SessionFactoryDep, SettingsDep, StoreDep
from app.db.models import Document, DocumentSource
from app.schemas.document import ArxivImport, DocumentOut, DocumentUpdate
from app.services.arxiv import base_id, download_arxiv_pdf, fetch_arxiv_title
from app.services.ingestion import ingest_document
from app.services.pdf import extract_title

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB


class Ingestor:
    """Bundles what the background pipeline needs, so routes take one parameter, not five."""

    def __init__(
        self,
        background: BackgroundTasks,
        repo: RepoDep,
        settings: SettingsDep,
        session_factory: SessionFactoryDep,
        embedder: EmbedderDep,
        store: StoreDep,
    ) -> None:
        self.background = background
        self.repo = repo
        self.settings = settings
        self.session_factory = session_factory
        self.embedder = embedder
        self.store = store

    def schedule(self, document_id: uuid.UUID) -> None:
        self.background.add_task(
            ingest_document,
            document_id,
            self.session_factory,
            self.embedder,
            self.store,
            self.settings,
        )

    def register_pdf(
        self,
        data: bytes,
        source: DocumentSource,
        source_ref: str,
        response: Response,
        title: str | None = None,
    ) -> Document:
        if not data.startswith(b"%PDF"):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "file is not a PDF")
        if len(data) > MAX_PDF_BYTES:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "PDF larger than 50 MB")

        # Idempotency #1: the same bytes always give the same hash -> return the existing one.
        content_hash = hashlib.sha256(data).hexdigest()
        existing = self.repo.get_by_hash(content_hash)
        if existing is not None:
            response.status_code = status.HTTP_200_OK
            return existing

        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)
        path = self.settings.upload_dir / f"{content_hash}.pdf"
        path.write_bytes(data)

        document = self.repo.add(
            Document(
                title=title or extract_title(path, fallback=source_ref),
                source=source,
                source_ref=source_ref,
                content_hash=content_hash,
                file_path=str(path),
            )
        )
        self.schedule(document.id)
        response.status_code = status.HTTP_201_CREATED
        return document


IngestorDep = Annotated[Ingestor, Depends()]


@router.post("", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile,
    response: Response,
    ingestor: IngestorDep,
    title: Annotated[str | None, Form(max_length=500)] = None,
) -> Document:
    """Upload a PDF. `title` is optional; without it we guess from the PDF."""
    data = await file.read()
    return ingestor.register_pdf(
        data, DocumentSource.UPLOAD, file.filename or "upload.pdf", response, title=title
    )


@router.post("/arxiv", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
def import_arxiv(body: ArxivImport, response: Response, ingestor: IngestorDep) -> Document:
    """Import by arXiv id. The title comes from arXiv's metadata."""
    paper_id = base_id(body.arxiv_id)

    # Idempotency #2: same paper id -> same document, even if arXiv serves different bytes.
    existing = ingestor.repo.get_by_source(DocumentSource.ARXIV, paper_id)
    if existing is not None:
        response.status_code = status.HTTP_200_OK
        return existing

    try:
        data = download_arxiv_pdf(body.arxiv_id)
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"arXiv download failed: {exc}") from exc
    title = fetch_arxiv_title(paper_id)
    return ingestor.register_pdf(data, DocumentSource.ARXIV, paper_id, response, title=title)


@router.get("", response_model=list[DocumentOut])
def list_documents(repo: RepoDep) -> list[Document]:
    return list(repo.list())


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: uuid.UUID, repo: RepoDep) -> Document:
    document = repo.get(document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    return document


@router.patch("/{document_id}", response_model=DocumentOut)
def update_document(
    document_id: uuid.UUID, body: DocumentUpdate, repo: RepoDep, store: StoreDep
) -> Document:
    """Fix a title. Updates PostgreSQL and the copy stored on each chunk in Qdrant."""
    document = get_document(document_id, repo)
    repo.rename(document, body.title)
    store.set_title(document.id, body.title)
    return document


@router.post("/{document_id}/reingest", response_model=DocumentOut, status_code=202)
def reingest_document(document_id: uuid.UUID, ingestor: IngestorDep) -> Document:
    """Run the pipeline again (e.g. after changing chunk_size). Never creates duplicates."""
    document = get_document(document_id, ingestor.repo)
    ingestor.schedule(document.id)
    return document
