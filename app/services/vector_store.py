"""Qdrant wrapper: store chunk vectors with their metadata (payload), and search them.

Idempotency trick: each point's id is derived from (document id, chunk index) with uuid5.
Re-ingesting the same document overwrites the same points instead of adding duplicates.
"""

import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.services.chunking import Chunk


@dataclass(frozen=True)
class ScoredChunk:
    document_id: uuid.UUID
    title: str
    page: int
    chunk_index: int
    score: float
    text: str


def point_id(document_id: uuid.UUID, chunk_index: int) -> str:
    return str(uuid.uuid5(document_id, str(chunk_index)))


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self.client = client
        self.collection = collection

    def ensure_collection(self, dimension: int) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            self.collection,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
        # An index on document_id makes "delete/filter by document" fast.
        self.client.create_payload_index(
            self.collection, "document_id", field_schema=PayloadSchemaType.KEYWORD
        )

    def delete_document(self, document_id: uuid.UUID) -> None:
        if not self.client.collection_exists(self.collection):
            return
        self.client.delete(
            self.collection,
            points_selector=FilterSelector(filter=_by_document(document_id)),
        )

    def upsert_chunks(
        self,
        document_id: uuid.UUID,
        title: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        points = [
            PointStruct(
                id=point_id(document_id, chunk.index),
                vector=vector,
                payload={
                    "document_id": str(document_id),
                    "title": title,
                    "page": chunk.page,
                    "chunk_index": chunk.index,
                    "text": chunk.text,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(self.collection, points=points)

    def set_title(self, document_id: uuid.UUID, title: str) -> None:
        """Update the title stored on every chunk of a document (no re-embedding needed)."""
        if not self.client.collection_exists(self.collection):
            return
        self.client.set_payload(
            self.collection, payload={"title": title}, points=_by_document(document_id)
        )

    def count(self, document_id: uuid.UUID | None = None) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        flt = _by_document(document_id) if document_id else None
        return self.client.count(self.collection, count_filter=flt, exact=True).count

    def search(self, vector: list[float], limit: int = 5) -> list[ScoredChunk]:
        if not self.client.collection_exists(self.collection):
            return []
        result = self.client.query_points(self.collection, query=vector, limit=limit)
        hits = []
        for p in result.points:
            payload = p.payload or {}
            hits.append(
                ScoredChunk(
                    document_id=uuid.UUID(payload["document_id"]),
                    title=payload["title"],
                    page=payload["page"],
                    chunk_index=payload["chunk_index"],
                    score=p.score,
                    text=payload["text"],
                )
            )
        return hits


def _by_document(document_id: uuid.UUID) -> Filter:
    return Filter(
        must=[FieldCondition(key="document_id", match=MatchValue(value=str(document_id)))]
    )
