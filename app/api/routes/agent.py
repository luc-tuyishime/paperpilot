"""POST /agent: a multi-step question answered by the agent, with memory per conversation.

Send `thread_id` from a previous response to continue that conversation.
"""

import logging
from dataclasses import asdict
from functools import lru_cache
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.agent.graph import build_agent, run_agent
from app.agent.runtime import build_chat_model, build_checkpointer
from app.agent.tools import make_tools
from app.api.deps import EmbedderDep, SessionFactoryDep, SettingsDep, StoreDep
from app.core.config import get_settings
from app.llm.base import LLMError
from app.schemas.agent import AgentRequest, AgentResponse

router = APIRouter(prefix="/agent", tags=["agent"])
logger = logging.getLogger("paperpilot.agent")


@lru_cache
def _chat_model() -> BaseChatModel:
    return build_chat_model(get_settings())


def get_chat_model() -> BaseChatModel:
    try:
        return _chat_model()
    except LLMError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


@lru_cache
def get_checkpointer() -> BaseCheckpointSaver[Any]:
    return build_checkpointer(get_settings().database_url)


@router.post("", response_model=AgentResponse)
def ask_agent(
    body: AgentRequest,
    settings: SettingsDep,
    embedder: EmbedderDep,
    store: StoreDep,
    session_factory: SessionFactoryDep,
    model: Annotated[BaseChatModel, Depends(get_chat_model)],
    checkpointer: Annotated[BaseCheckpointSaver[Any], Depends(get_checkpointer)],
) -> AgentResponse:
    tools = make_tools(embedder, store, session_factory, settings.min_relevance)
    graph = build_agent(model, tools, checkpointer)
    try:
        result = run_agent(graph, body.message, body.thread_id, settings.agent_max_steps)
    except Exception as exc:  # provider errors from LangChain have many types
        logger.exception("agent failed")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"agent failed: {exc}") from exc
    logger.info(
        "agent thread=%s steps=%d tools=%s tokens=%d/%d stopped=%s",
        result.thread_id,
        len(result.steps),
        [s.tool for s in result.steps],
        result.input_tokens,
        result.output_tokens,
        result.stopped_early,
    )
    return AgentResponse(**asdict(result))
