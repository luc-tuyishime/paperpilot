"""Run the evaluation set against a running PaperPilot API and write a report.

    uv run python -m evals.run_eval                       # local API on :8000
    uv run python -m evals.run_eval --ingest              # first import the papers from arXiv
    uv run python -m evals.run_eval --only transformer    # only questions whose id contains this

Exit code 1 if any metric is below its threshold: that's what makes CI fail.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx
import yaml

from evals.scoring import (
    METRICS,
    QuestionResult,
    arxiv_of,
    failures,
    score_question,
    summarize,
)


def ingest(client: httpx.Client, papers: list[str], timeout_s: int = 600) -> None:
    for paper in papers:
        r = client.post("/documents/arxiv", json={"arxiv_id": str(paper)})
        r.raise_for_status()
        print(f"  {paper}: {r.status_code} {r.json()['status']}")
        time.sleep(3)  # be polite to arXiv
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        docs = client.get("/documents").json()
        pending = [d for d in docs if d["status"] in ("pending", "processing")]
        failed = [d for d in docs if d["status"] == "failed"]
        if failed:
            sys.exit(f"ingestion failed: {[(d['source_ref'], d['error']) for d in failed]}")
        if not pending:
            print(f"  all {len(docs)} documents ready")
            return
        time.sleep(5)
    sys.exit("ingestion timed out")


def ask(client: httpx.Client, question: str) -> dict[str, Any]:
    for attempt in range(3):
        r = client.post("/ask", json={"question": question})
        if r.status_code in (429, 502, 503) and attempt < 2:
            print(f"    {r.status_code}, waiting 30s (free-tier rate limit?)")
            time.sleep(30)
            continue
        r.raise_for_status()
        result: dict[str, Any] = r.json()
        return result
    raise RuntimeError("unreachable")


def mark(value: bool | None) -> str:
    return "–" if value is None else ("✅" if value else "❌")


def report(
    results: list[QuestionResult], summary: dict[str, float], thresholds: dict[str, float]
) -> str:
    lines = ["# PaperPilot evaluation", "", "| Metric | Score | Threshold | |", "|---|---|---|---|"]
    for m in METRICS:
        t = thresholds.get(m, 0.0)
        lines.append(f"| {m} | {summary[m]:.2f} | {t:.2f} | {mark(summary[m] >= t)} |")
    answered = [r for r in results if r.latency_ms]
    if answered:
        avg_latency = sum(r.latency_ms for r in answered) // len(answered)
        tokens = sum(r.input_tokens + r.output_tokens for r in results)
        lines += [
            "",
            f"{len(results)} questions · avg LLM latency {avg_latency} ms · {tokens} tokens",
        ]
    lines += [
        "",
        "| Question | Retrieved | Cited | Correct | Refusal | Best score |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| `{r.id}` | {mark(r.retrieval_hit)} | {mark(r.citation_hit)} "
            f"| {mark(r.answer_correct)} | {mark(r.refusal_correct)} | {r.best_score:.3f} |"
        )
    bad = [r for r in results if not r.refusal_correct or r.answer_correct is False]
    if bad:
        lines += ["", "## Answers to look at", ""]
        for r in bad:
            lines += [f"**{r.id}**: {r.question}", "", f"> {r.answer[:600]}", ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--questions", default="evals/questions.yaml")
    parser.add_argument("--ingest", action="store_true", help="import the papers from arXiv first")
    parser.add_argument("--only", help="run only questions whose id contains this text")
    parser.add_argument("--delay", type=float, default=7.0, help="seconds between questions")
    parser.add_argument("--out", default="evals/results")
    args = parser.parse_args()

    spec = yaml.safe_load(Path(args.questions).read_text())
    questions = [q for q in spec["questions"] if not args.only or args.only in q["id"]]
    thresholds: dict[str, float] = spec.get("thresholds", {})

    with httpx.Client(base_url=args.base_url, timeout=180) as client:
        if args.ingest:
            print("Importing papers...")
            ingest(client, spec["papers"])
        doc_to_arxiv = {d["id"]: arxiv_of(d) for d in client.get("/documents").json()}

        results: list[QuestionResult] = []
        for i, q in enumerate(questions, start=1):
            print(f"[{i}/{len(questions)}] {q['id']}")
            response = ask(client, q["question"])
            result = score_question(q, response, doc_to_arxiv)
            results.append(result)
            print(
                f"    correct={mark(result.answer_correct)} refusal={mark(result.refusal_correct)}"
            )
            if response.get("usage") and i < len(questions):
                time.sleep(args.delay)  # stay under the free tier's requests-per-minute

    summary = summarize(results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(report(results, summary, thresholds))
    (out / "results.json").write_text(
        json.dumps({"summary": summary, "results": [asdict(r) for r in results]}, indent=2)
    )
    print("\n" + json.dumps(summary, indent=2))
    print(f"Report: {out / 'report.md'}")

    problems = failures(summary, thresholds) if not args.only else []
    if problems:
        print("\nFAILED thresholds:\n  " + "\n  ".join(problems))
        sys.exit(1)


if __name__ == "__main__":
    main()
