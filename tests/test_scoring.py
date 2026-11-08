from typing import Any

from evals.scoring import arxiv_of, contains_all, failures, score_question, summarize

DOC_TRANSFORMER = "11111111-1111-1111-1111-111111111111"
DOC_GAN = "22222222-2222-2222-2222-222222222222"
DOCS = {DOC_TRANSFORMER: "1706.03762", DOC_GAN: "1406.2661"}


def response(answer: str, grounded: bool, retrieved: list[str], cited: list[str]) -> dict[str, Any]:
    return {
        "answer": answer,
        "grounded": grounded,
        "best_score": 0.7,
        "sources": [{"document_id": d} for d in retrieved],
        "citations": [{"document_id": d} for d in cited],
        "usage": {"latency_ms": 900, "input_tokens": 500, "output_tokens": 30},
    }


def test_arxiv_of_reads_imports_and_file_names() -> None:
    assert arxiv_of({"source_ref": "1706.03762"}) == "1706.03762"
    assert arxiv_of({"source_ref": "1706.03762v7.pdf"}) == "1706.03762"
    assert arxiv_of({"source_ref": "my-notes.pdf"}) is None


def test_contains_all_groups_and_alternatives() -> None:
    answer = "The generator plays a minimax game against the discriminator."
    assert contains_all(answer, [["generator", "generative"], ["discriminat"]])
    assert not contains_all(answer, [["generator"], ["Cora"]])


def test_good_answer_scores_everything() -> None:
    spec = {"id": "q", "question": "?", "arxiv": "1706.03762", "must_include": [["28.4"]]}
    r = score_question(
        spec, response("28.4 BLEU [1].", True, [DOC_TRANSFORMER, DOC_GAN], [DOC_TRANSFORMER]), DOCS
    )
    assert (r.retrieval_hit, r.citation_hit, r.answer_correct, r.refusal_correct) == (
        True,
        True,
        True,
        True,
    )


def test_wrong_paper_cited_is_caught() -> None:
    spec = {"id": "q", "question": "?", "arxiv": "1706.03762", "must_include": [["28.4"]]}
    r = score_question(spec, response("28.4 [1].", True, [DOC_GAN], [DOC_GAN]), DOCS)
    assert r.retrieval_hit is False and r.citation_hit is False


def test_unanswerable_question() -> None:
    spec = {"id": "q", "question": "capital of Rwanda?", "answerable": False}
    refused = score_question(spec, response("I don't know...", False, [], []), DOCS)
    invented = score_question(spec, response("Kigali [1].", True, [DOC_GAN], [DOC_GAN]), DOCS)
    assert refused.refusal_correct and refused.answer_correct is None
    assert not invented.refusal_correct  # a hallucination is caught


def test_summary_ignores_not_applicable_and_checks_thresholds() -> None:
    good = {"id": "a", "question": "?", "arxiv": "1706.03762", "must_include": [["28.4"]]}
    unanswerable = {"id": "b", "question": "?", "answerable": False}
    results = [
        score_question(
            good, response("28.4 [1]", True, [DOC_TRANSFORMER], [DOC_TRANSFORMER]), DOCS
        ),
        score_question(unanswerable, response("Kigali", True, [], []), DOCS),
    ]
    summary = summarize(results)
    assert summary["answer_correct"] == 1.0  # only the answerable question counts
    assert summary["refusal_correct"] == 0.5
    assert failures(summary, {"refusal_correct": 0.8}) == ["refusal_correct: 0.50 < 0.80"]
