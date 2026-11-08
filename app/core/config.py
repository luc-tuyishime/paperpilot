"""Application settings, read from environment variables (or a .env file).

Every setting has a safe default so the app runs with zero configuration.
In Docker or Kubernetes we override them with environment variables,
never by editing code.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="PAPERPILOT_")

    app_name: str = "PaperPilot"
    environment: str = "local"  # local | docker | ci | production
    version: str = "0.3.0"

    # PostgreSQL: the source of truth for documents and their status
    database_url: str = "postgresql+psycopg://paperpilot:paperpilot@localhost:5432/paperpilot"

    # Qdrant: where the chunk vectors live
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "chunks"

    # Ollama: local embeddings (runs natively on your Mac, not in Docker)
    ollama_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    embed_batch_size: int = 32

    # Chunking (in characters). ~1000 chars ≈ 200-250 tokens.
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Where uploaded PDFs are stored on disk
    upload_dir: Path = Path("data/uploads")

    # LLM that writes the answers: "gemini" (free tier, fast) or "ollama" (local, offline)
    llm_provider: Literal["gemini", "ollama"] = "gemini"
    gemini_api_key: SecretStr | None = None  # SecretStr: never printed in logs or errors
    gemini_model: str = "gemini-2.5-flash"
    ollama_chat_model: str = "llama3.2:3b"
    llm_temperature: float = 0.1  # low = factual, repeatable answers

    # Retrieval for RAG
    retrieval_top_k: int = 6  # how many chunks go into the prompt
    # Chunks below this cosine score are ignored. If none pass -> "I don't know" with no LLM call.
    # Tune it with the evaluation set (Day 2), don't guess.
    min_relevance: float = 0.55


@lru_cache
def get_settings() -> Settings:
    """Build Settings once and reuse it (cached), so .env is read only one time."""
    return Settings()
