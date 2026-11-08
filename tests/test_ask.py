import json
from collections.abc import Callable

from fastapi.testclient import TestClient

from app.services.rag import IDK, cited_numbers, is_refusal
from tests.conftest import FakeLLM

TRANSFORMER = "The Transformer big model achieves 28.4 BLEU on English to German translation. " * 6
GAN = "A generator is trained against a discriminator in a minimax game. " * 6


def ingest(client: TestClient, make_pdf: Callable[..., bytes]) -> None:
    pdf = make_pdf([TRANSFORMER, GAN])
    client.post("/documents", files={"file": ("p.pdf", pdf, "application/pdf")})


def test_cited_numbers_parses_and_validates() -> None:
    assert cited_numbers("A [1]. B [2][3]. C [1, 4]. D [9].", n_sources=4) == [1, 2, 3, 4]
    assert cited_numbers("no citations", n_sources=3) == []


def test_is_refusal() -> None:
    assert is_refusal(IDK)
    assert is_refusal("  i don't know.")
    assert not is_refusal("The answer is 28.4 [1].")


def test_ask_answers_with_citations(
    client: TestClient, make_pdf: Callable[..., bytes], llm: FakeLLM
) -> None:
    ingest(client, make_pdf)

    body = client.post(
        "/ask", json={"question": "What BLEU score does the Transformer get?"}
    ).json()

    assert body["grounded"] is True
    assert "[1]" in body["answer"]
    assert body["citations"][0]["number"] == 1
    assert body["citations"][0]["page"] == 1  # the Transformer text is on page 1
    assert body["usage"]["provider"] == "fake"
    # the prompt numbered the sources and contained the question
    assert '<source id="1"' in llm.prompts[0] and "BLEU" in llm.prompts[0]


def test_unrelated_question_refuses_without_calling_the_llm(
    client: TestClient, make_pdf: Callable[..., bytes], llm: FakeLLM
) -> None:
    ingest(client, make_pdf)

    body = client.post("/ask", json={"question": "What is the capital of Rwanda?"}).json()

    assert body["answer"] == IDK
    assert body["grounded"] is False
    assert body["usage"] is None
    assert llm.prompts == []  # no tokens spent


def test_model_refusal_is_reported_as_not_grounded(
    client: TestClient, make_pdf: Callable[..., bytes], llm: FakeLLM
) -> None:
    ingest(client, make_pdf)
    llm.reply = IDK

    body = client.post(
        "/ask", json={"question": "What BLEU score does the Transformer get?"}
    ).json()

    assert body["grounded"] is False
    assert body["citations"] == []


def test_question_length_is_validated(client: TestClient) -> None:
    assert client.post("/ask", json={"question": "x" * 1001}).status_code == 422


def test_stream_sends_sources_tokens_then_done(
    client: TestClient, make_pdf: Callable[..., bytes]
) -> None:
    ingest(client, make_pdf)

    with client.stream(
        "POST", "/ask/stream", json={"question": "how is the generator trained"}
    ) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        raw = r.read().decode()

    events = [
        (block.split("\n")[0].removeprefix("event: "), json.loads(block.split("\n")[1][6:]))
        for block in raw.strip().split("\n\n")
    ]
    names = [name for name, _ in events]
    assert names[0] == "sources" and names[-1] == "done"
    assert set(names[1:-1]) == {"token"}
    assert events[0][1][0]["page"] == 2  # the GAN text is on page 2
    answer = "".join(data["text"] for name, data in events if name == "token")
    assert "generator" in answer.lower()
    assert events[-1][1]["grounded"] is True and events[-1][1]["cited"] == [1]
