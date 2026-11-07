"""Test setup: the real app, with every external system replaced by a fast fake.

- PostgreSQL -> in-memory SQLite (same SQLAlchemy code, no server)
- Qdrant     -> QdrantClient(":memory:") (the real client, running in-process)
- Ollama     -> FakeEmbedder (bag-of-words vectors, deterministic, no network)
"""

import hashlib
import math
import re
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from fpdf import FPDF
from qdrant_client import QdrantClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.deps import get_embedder, get_vector_store
from app.core.config import Settings, get_settings
from app.db.session import Base, get_db, get_session_factory
from app.main import create_app
from app.services.vector_store import VectorStore

DIM = 64


class FakeEmbedder:
    """Each word adds 1 to one of 64 slots. Texts sharing words get similar vectors,
    so search results are meaningful in tests, without any model."""

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for word in re.findall(r"[a-z]+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture
def store() -> VectorStore:
    return VectorStore(QdrantClient(":memory:"), "test_chunks")


@pytest.fixture
def session_factory() -> sessionmaker[Session]:
    # StaticPool: every session shares ONE in-memory connection, so they see the same tables.
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def client(
    tmp_path: Path, store: VectorStore, session_factory: sessionmaker[Session]
) -> Iterator[TestClient]:
    app = create_app()
    settings = Settings(upload_dir=tmp_path / "uploads", chunk_size=300, chunk_overlap=50)

    def _db() -> Iterator[Session]:
        with session_factory() as s:
            yield s

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_session_factory] = lambda: session_factory
    app.dependency_overrides[get_embedder] = FakeEmbedder
    app.dependency_overrides[get_vector_store] = lambda: store
    with TestClient(app) as c:
        yield c


@pytest.fixture
def make_pdf() -> Callable[[list[str]], bytes]:
    """Build a real PDF in memory: one string per page."""

    def _make(pages: list[str], title: str = "A Test Paper") -> bytes:
        pdf = FPDF()
        pdf.set_title(title)
        pdf.set_font("Helvetica", size=11)
        for text in pages:
            pdf.add_page()
            pdf.multi_cell(0, 6, text)
        return bytes(pdf.output())

    return _make
