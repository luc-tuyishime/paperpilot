"""Shared dependencies. Each one is a single place tests can override."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.db.session import get_db, get_session_factory
from app.repositories.documents import DocumentRepository
from app.services.embeddings import Embedder, OllamaEmbedder
from app.services.vector_store import VectorStore


@lru_cache
def get_embedder() -> Embedder:
    s = get_settings()
    return OllamaEmbedder(base_url=s.ollama_url, model=s.embed_model)


@lru_cache
def get_vector_store() -> VectorStore:
    s = get_settings()
    return VectorStore(QdrantClient(url=s.qdrant_url), s.qdrant_collection)


def get_document_repo(db: Annotated[Session, Depends(get_db)]) -> DocumentRepository:
    return DocumentRepository(db)


SettingsDep = Annotated[Settings, Depends(get_settings)]
RepoDep = Annotated[DocumentRepository, Depends(get_document_repo)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
StoreDep = Annotated[VectorStore, Depends(get_vector_store)]
SessionFactoryDep = Annotated[sessionmaker[Session], Depends(get_session_factory)]
