"""What the agent runs on: a chat model that can call tools, and a checkpointer for memory.

The RAG endpoint (Day 2) uses our own small LLM clients. The agent uses LangChain chat models
instead, because tool calling (the model asking to run a function) has a different format
for every provider, and LangChain already handles it.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.core.config import Settings
from app.llm.base import LLMError


def build_chat_model(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "gemini":
        if settings.gemini_api_key is None:
            raise LLMError("PAPERPILOT_GEMINI_API_KEY is not set (or use LLM_PROVIDER=ollama)")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=settings.llm_temperature,
            max_retries=3,
        )
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.ollama_chat_model,
        base_url=settings.ollama_url,
        temperature=settings.llm_temperature,
    )


_pools: list[ConnectionPool[Any]] = []


def close_pools() -> None:
    """Called when the server stops, so connections close cleanly."""
    while _pools:
        _pools.pop().close()


def build_checkpointer(database_url: str) -> BaseCheckpointSaver[Any]:
    """Conversation memory in PostgreSQL. LangGraph creates its own tables on setup()."""
    conninfo = database_url.replace("postgresql+psycopg://", "postgresql://")
    pool: ConnectionPool[Any] = ConnectionPool(
        conninfo,
        max_size=5,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=True,
    )
    _pools.append(pool)
    saver = PostgresSaver(pool)
    saver.setup()
    return saver
