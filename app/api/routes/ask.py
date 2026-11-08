"""Ask a question about the indexed papers.

POST /ask         -> one JSON response (simple to test and to evaluate)
POST /ask/stream  -> Server-Sent Events: the answer appears word by word, like ChatGPT

SSE events, in order:
  event: sources  data: [ {number, title, page, score, ...}, ... ]   (before the answer)
  event: token    data: {"text": "..."}                              (many times)
  event: done     data: {"grounded": true, "cited": [1, 3], "usage": {...}}
  event: error    data: {"detail": "..."}                            (only if the LLM fails)
"""

import json
import logging
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.deps import EmbedderDep, LLMDep, SettingsDep, StoreDep
from app.llm.base import LLMError, Usage
from app.schemas.ask import AskRequest, AskResponse, Source, UsageOut
from app.services.rag import (
    IDK,
    SYSTEM_PROMPT,
    Retrieval,
    build_prompt,
    cited_numbers,
    is_refusal,
    retrieve,
)

router = APIRouter(prefix="/ask", tags=["ask"])
logger = logging.getLogger("paperpilot.ask")


def _sources(retrieval: Retrieval) -> list[Source]:
    return [
        Source(
            number=i,
            document_id=s.document_id,
            title=s.title,
            page=s.page,
            score=round(s.score, 4),
            text=s.text,
        )
        for i, s in enumerate(retrieval.sources, start=1)
    ]


def _usage_out(usage: Usage) -> UsageOut:
    return UsageOut(
        provider=usage.provider,
        model=usage.model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        latency_ms=usage.latency_ms,
    )


def _log(question: str, retrieval: Retrieval, grounded: bool, usage: Usage | None) -> None:
    # One structured line per question: the raw material for metrics and dashboards.
    logger.info(
        "ask %s",
        json.dumps(
            {
                "question_chars": len(question),
                "best_score": round(retrieval.best_score, 4),
                "n_sources": len(retrieval.sources),
                "grounded": grounded,
                "input_tokens": usage.input_tokens if usage else 0,
                "output_tokens": usage.output_tokens if usage else 0,
                "latency_ms": usage.latency_ms if usage else 0,
            }
        ),
    )


@router.post("", response_model=AskResponse)
def ask(
    body: AskRequest, settings: SettingsDep, embedder: EmbedderDep, store: StoreDep, llm: LLMDep
) -> AskResponse:
    retrieval = retrieve(
        body.question,
        embedder,
        store,
        top_k=body.top_k or settings.retrieval_top_k,
        min_relevance=settings.min_relevance,
    )
    sources = _sources(retrieval)

    if not retrieval.grounded:  # nothing relevant: refuse without paying for an LLM call
        _log(body.question, retrieval, False, None)
        return AskResponse(
            answer=IDK,
            grounded=False,
            citations=[],
            sources=[],
            best_score=retrieval.best_score,
            usage=None,
        )

    try:
        result = llm.generate(SYSTEM_PROMPT, build_prompt(body.question, retrieval.sources))
    except LLMError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

    cited = cited_numbers(result.text, len(sources))
    grounded = not is_refusal(result.text)
    _log(body.question, retrieval, grounded, result.usage)
    return AskResponse(
        answer=result.text.strip(),
        grounded=grounded,
        citations=[s for s in sources if s.number in cited],
        sources=sources,
        best_score=retrieval.best_score,
        usage=_usage_out(result.usage),
    )


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("/stream")
def ask_stream(
    body: AskRequest, settings: SettingsDep, embedder: EmbedderDep, store: StoreDep, llm: LLMDep
) -> StreamingResponse:
    retrieval = retrieve(
        body.question,
        embedder,
        store,
        top_k=body.top_k or settings.retrieval_top_k,
        min_relevance=settings.min_relevance,
    )
    sources = _sources(retrieval)

    def events() -> Iterator[str]:
        yield _sse("sources", [s.model_dump() for s in sources])
        if not retrieval.grounded:
            _log(body.question, retrieval, False, None)
            yield _sse("token", {"text": IDK})
            yield _sse("done", {"grounded": False, "cited": [], "usage": None})
            return

        usage = Usage()
        answer: list[str] = []
        try:
            prompt = build_prompt(body.question, retrieval.sources)
            for piece in llm.stream(SYSTEM_PROMPT, prompt, usage):
                answer.append(piece)
                yield _sse("token", {"text": piece})
        except LLMError as exc:
            yield _sse("error", {"detail": str(exc)})
            return

        text = "".join(answer)
        grounded = not is_refusal(text)
        _log(body.question, retrieval, grounded, usage)
        yield _sse(
            "done",
            {
                "grounded": grounded,
                "cited": cited_numbers(text, len(sources)),
                "usage": _usage_out(usage).model_dump(),
            },
        )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},  # no proxy buffering
    )
