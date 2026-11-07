"""Embeddings: text -> vector of numbers. Similar meaning => nearby vectors.

`Embedder` is a Protocol (an interface). The app depends on the interface, so we can
swap Ollama for Gemini or OpenAI later, and tests use a fake one with no network.
"""

from typing import Protocol

import httpx


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class OllamaEmbedder:
    # nomic-embed-text is trained with task prefixes: it works better when told whether
    # a text is something to be found ("search_document") or a question ("search_query").
    DOC_PREFIX = "search_document: "
    QUERY_PREFIX = "search_query: "

    def __init__(self, base_url: str, model: str, timeout: float = 120.0) -> None:
        self.model = model
        self.client = httpx.Client(base_url=base_url, timeout=timeout)

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        response = self.client.post("/api/embed", json={"model": self.model, "input": inputs})
        response.raise_for_status()
        embeddings: list[list[float]] = response.json()["embeddings"]
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([self.DOC_PREFIX + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.QUERY_PREFIX + text])[0]
