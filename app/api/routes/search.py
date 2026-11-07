"""Semantic search over all indexed chunks. Day 2 builds the RAG /ask endpoint on top of this."""

from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import EmbedderDep, StoreDep
from app.schemas.document import SearchHit

router = APIRouter(tags=["search"])


@router.get("/search", response_model=list[SearchHit])
def search(
    q: Annotated[str, Query(min_length=2, max_length=500)],
    embedder: EmbedderDep,
    store: StoreDep,
    limit: Annotated[int, Query(ge=1, le=20)] = 5,
) -> list[SearchHit]:
    vector = embedder.embed_query(q)
    return [SearchHit(**vars(hit)) for hit in store.search(vector, limit=limit)]
