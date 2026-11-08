"""Scoring functions for the evaluation. Pure functions: no network, easy to unit test.

Four metrics, each a fraction between 0 and 1:
- retrieval_hit:   was the right paper among the retrieved sources?    (search quality)
- citation_hit:    did the answer cite the right paper?                (grounding)
- answer_correct:  does the answer contain the expected facts?         (answer quality)
- refusal_correct: did it say "I don't know" exactly when it should?   (no hallucination)

These are cheap, deterministic checks. An LLM judge (Ragas "faithfulness", "answer relevancy")
measures more nuance, but costs API calls and varies between runs. Start here.
"""

import re
from dataclasses import dataclass, field
from typing import Any

ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5})")


def arxiv_of(document: dict[str, Any]) -> str | None:
    """Find the arXiv id of a document: the source_ref of an arXiv import, or a file name
    like '1706.03762.pdf' for an upload."""
    match = ARXIV_ID.search(document.get("source_ref", ""))
    return match.group(1) if match else None


def contains_all(answer: str, must_include: list[list[str]]) -> bool:
    text = answer.lower()
    return all(any(alt.lower() in text for alt in group) for group in must_include)


@dataclass
class QuestionResult:
    id: str
    question: str
    answerable: bool
    answer: str
    refused: bool
    best_score: float
    latency_ms: int
    input_tokens: int
    output_tokens: int
    retrieval_hit: bool | None = None  # None = not applicable (unanswerable question)
    citation_hit: bool | None = None
    answer_correct: bool | None = None
    refusal_correct: bool = False
    cited_papers: list[str] = field(default_factory=list)


def score_question(
    spec: dict[str, Any], response: dict[str, Any], doc_to_arxiv: dict[str, str | None]
) -> QuestionResult:
    answerable = spec.get("answerable", True)
    refused = not response["grounded"]
    usage = response.get("usage") or {}

    def papers(sources: list[dict[str, Any]]) -> list[str]:
        return [p for s in sources if (p := doc_to_arxiv.get(str(s["document_id"])))]

    retrieved = papers(response["sources"])
    cited = papers(response["citations"])
    result = QuestionResult(
        id=spec["id"],
        question=spec["question"],
        answerable=answerable,
        answer=response["answer"],
        refused=refused,
        best_score=response["best_score"],
        latency_ms=usage.get("latency_ms", 0),
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        refusal_correct=(refused != answerable),
        cited_papers=sorted(set(cited)),
    )
    if answerable:
        expected = str(spec["arxiv"])
        result.retrieval_hit = expected in retrieved
        result.citation_hit = expected in cited
        result.answer_correct = contains_all(response["answer"], spec["must_include"])
    return result


METRICS = ("retrieval_hit", "citation_hit", "answer_correct", "refusal_correct")


def summarize(results: list[QuestionResult]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric in METRICS:
        values = [v for r in results if (v := getattr(r, metric)) is not None]
        summary[metric] = round(sum(values) / len(values), 3) if values else 0.0
    return summary


def failures(summary: dict[str, float], thresholds: dict[str, float]) -> list[str]:
    return [
        f"{m}: {summary[m]:.2f} < {t:.2f}" for m, t in thresholds.items() if summary.get(m, 0.0) < t
    ]
