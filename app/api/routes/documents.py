"""Document endpoints: upload a PDF, import from arXiv, list, get, re-ingest.

Every new document is saved as PENDING and the response returns immediately (201).
The slow part (extract, embed, index) runs afterwards as a background task.
Clients poll GET /documents/{id} to watch the status move to READY.
"""

import hashlib
import uuid

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Response, UploadFile, status

from app.api.deps import EmbedderDep, RepoDep, SessionFactoryDep, SettingsDep, StoreDep
from app.db.models import Document, DocumentSource
from app.schemas.document import ArxivImport, DocumentOut
from app.services.arxiv import download_arxiv_pdf
from app.services.ingestion import ingest_document
from app.services.pdf import extract_title

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB


def _register_pdf(
    data: bytes,
    source: DocumentSource,
    source_ref: str,
    response: Response,
    background: BackgroundTasks,
    repo: RepoDep,
    settings: SettingsDep,
    session_factory: SessionFactoryDep,
    embedder: EmbedderDep,
    store: StoreDep,
) -> Document:
    if not data.startswith(b"%PDF"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "file is not a PDF")
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "PDF larger than 50 MB")

    # Idempotency: the same bytes always give the same hash -> return the existing document.
    content_hash = hashlib.sha256(data).hexdigest()
    existing = repo.get_by_hash(content_hash)
    if existing is not None:
        response.status_code = status.HTTP_200_OK
        return existing

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    path = settings.upload_dir / f"{content_hash}.pdf"
    path.write_bytes(data)

    document = repo.add(
        Document(
            title=extract_title(path, fallback=source_ref),
            source=source,
            source_ref=source_ref,
            content_hash=content_hash,
            file_path=str(path),
        )
    )
    background.add_task(ingest_document, document.id, session_factory, embedder, store, settings)
    response.status_code = status.HTTP_201_CREATED
    return document


@router.post("", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile,
    response: Response,
    background: BackgroundTasks,
    repo: RepoDep,
    settings: SettingsDep,
    session_factory: SessionFactoryDep,
    embedder: EmbedderDep,
    store: StoreDep,
) -> Document:
    data = await file.read()
    return _register_pdf(
        data, DocumentSource.UPLOAD, file.filename or "upload.pdf",
        response, background, repo, settings, session_factory, embedder, store,
    )  # fmt: skip


@router.post("/arxiv", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
def import_arxiv(
    body: ArxivImport,
    response: Response,
    background: BackgroundTasks,
    repo: RepoDep,
    settings: SettingsDep,
    session_factory: SessionFactoryDep,
    embedder: EmbedderDep,
    store: StoreDep,
) -> Document:
    try:
        data = download_arxiv_pdf(body.arxiv_id)
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"arXiv download failed: {exc}") from exc
    return _register_pdf(
        data, DocumentSource.ARXIV, body.arxiv_id,
        response, background, repo, settings, session_factory, embedder, store,
    )  # fmt: skip


@router.get("", response_model=list[DocumentOut])
def list_documents(repo: RepoDep) -> list[Document]:
    return list(repo.list())


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: uuid.UUID, repo: RepoDep) -> Document:
    document = repo.get(document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    return document


@router.post("/{document_id}/reingest", response_model=DocumentOut, status_code=202)
def reingest_document(
    document_id: uuid.UUID,
    background: BackgroundTasks,
    repo: RepoDep,
    settings: SettingsDep,
    session_factory: SessionFactoryDep,
    embedder: EmbedderDep,
    store: StoreDep,
) -> Document:
    """Run the pipeline again (e.g. after changing chunk_size). Never creates duplicates."""
    document = get_document(document_id, repo)
    background.add_task(ingest_document, document.id, session_factory, embedder, store, settings)
    return document
