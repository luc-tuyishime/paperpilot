"""The agent's tools. A tool = a Python function the LLM can decide to call.

The docstring and the argument names are what the LLM sees: it reads them to decide WHEN
to call the tool and WITH WHAT. So they're written for the model, as instructions.

Both tools reuse Day 1 code (PostgreSQL repository, embedder, Qdrant). The agent adds the
decision-making on top: which tool, how many times, with which query.
"""

from langchain_core.tools import BaseTool, tool
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import DocumentStatus
from app.repositories.documents import DocumentRepository
from app.services.embeddings import Embedder
from app.services.vector_store import VectorStore


def make_tools(
    embedder: Embedder,
    store: VectorStore,
    session_factory: sessionmaker[Session],
    min_relevance: float,
) -> list[BaseTool]:
    @tool
    def list_papers() -> str:
        """List the papers in the library (titles and arXiv ids). Use it to know what exists
        before searching, or when the user refers to a paper by a vague name."""
        with session_factory() as session:
            docs = DocumentRepository(session).list()
        ready = [d for d in docs if d.status == DocumentStatus.READY]
        if not ready:
            return "The library is empty."
        return "\n".join(f"- {d.title} (id: {d.source_ref})" for d in ready)

    @tool
    def search_papers(query: str) -> str:
        """Search the papers for passages relevant to `query`. Returns numbered passages with
        title and page. Call it once per paper or sub-question; use specific queries, e.g.
        'Transformer BLEU English-German' rather than 'results'."""
        hits = [
            h
            for h in store.search(embedder.embed_query(query), limit=5)
            if h.score >= min_relevance
        ]
        if not hits:
            return "No relevant passages found."
        return "\n\n".join(
            f"[{h.title}, p. {h.page}] (score {h.score:.2f})\n{h.text}" for h in hits
        )

    return [list_papers, search_papers]
