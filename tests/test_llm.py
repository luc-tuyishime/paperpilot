"""The Gemini client against fake HTTP responses shaped like the real API."""

import json

import httpx
import pytest

from app.llm import gemini
from app.llm.base import LLMError, Usage
from app.llm.gemini import GeminiLLM

ANSWER = {
    "candidates": [{"content": {"parts": [{"text": "28.4 BLEU [1]."}]}}],
    "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 8, "thoughtsTokenCount": 30},
}


def make(handler: httpx.MockTransport) -> GeminiLLM:
    return GeminiLLM(api_key="test-key", model="gemini-test", transport=handler)


def test_generate_parses_text_and_usage() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=ANSWER)

    result = make(httpx.MockTransport(handler)).generate("system", "question")

    assert result.text == "28.4 BLEU [1]."
    assert result.usage.input_tokens == 120
    assert result.usage.output_tokens == 38  # answer + hidden thinking tokens
    assert seen[0].headers["x-goog-api-key"] == "test-key"
    assert "key=" not in str(seen[0].url)  # key never in the URL
    body = json.loads(seen[0].content)
    assert body["systemInstruction"]["parts"][0]["text"] == "system"


def test_generate_retries_on_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gemini.time, "sleep", lambda _s: None)
    responses = iter([httpx.Response(429), httpx.Response(429), httpx.Response(200, json=ANSWER)])

    result = make(httpx.MockTransport(lambda _r: next(responses))).generate("s", "q")

    assert result.text.startswith("28.4")


def test_generate_gives_up_with_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gemini.time, "sleep", lambda _s: None)
    llm = make(httpx.MockTransport(lambda _r: httpx.Response(429, text="quota exceeded")))

    with pytest.raises(LLMError, match="429"):
        llm.generate("s", "q")


def test_stream_yields_pieces_and_fills_usage() -> None:
    events = [
        {"candidates": [{"content": {"parts": [{"text": "The big model "}]}}]},
        {
            "candidates": [{"content": {"parts": [{"text": "gets 28.4 [1]."}]}}],
            "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 9},
        },
    ]
    sse = "".join(f"data: {json.dumps(e)}\r\n\r\n" for e in events)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["alt"] == "sse"
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    usage = Usage()
    pieces = list(make(httpx.MockTransport(handler)).stream("s", "q", usage))

    assert pieces == ["The big model ", "gets 28.4 [1]."]
    assert (usage.input_tokens, usage.output_tokens) == (100, 9)
    assert usage.provider == "gemini"
