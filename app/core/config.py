"""Application settings, read from environment variables (or a .env file).

Every setting has a safe default so the app runs with zero configuration.
In Docker or Kubernetes we override them with environment variables,
never by editing code.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="PAPERPILOT_")

    app_name: str = "PaperPilot"
    environment: str = "local"  # local | docker | ci | production
    version: str = "0.2.0"

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


@lru_cache
def get_settings() -> Settings:
    """Build Settings once and reuse it (cached), so .env is read only one time."""
    return Settings()
