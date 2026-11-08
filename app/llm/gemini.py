"""Google Gemini over plain REST (no SDK): easier to read, fewer dependencies.

Docs: https://ai.google.dev/api/generate-content
"""

import json
import logging
import time
from collections.abc import Iterator
from typing import Any

import httpx

from app.llm.base import LLMError, LLMResult, Usage

logger = logging.getLogger(__name__)

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
RETRYABLE = {429, 500, 503}  # rate limited / temporary server errors


class GeminiLLM:
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        timeout: float = 60.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,  # tests inject a fake transport
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        # The key goes in a header, never in the URL (URLs end up in logs).
        self.client = httpx.Client(
            base_url=BASE_URL,
            headers={"x-goog-api-key": api_key},
            timeout=timeout,
            transport=transport,
        )

    def _body(self, system: str, prompt: str) -> dict[str, Any]:
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature},
        }

    def _usage(self, usage: Usage, meta: dict[str, Any]) -> None:
        usage.input_tokens = meta.get("promptTokenCount", 0)
        # "thoughts" are the hidden reasoning tokens of thinking models: billed as output.
        usage.output_tokens = meta.get("candidatesTokenCount", 0) + meta.get(
            "thoughtsTokenCount", 0
        )

    @staticmethod
    def _text(chunk: dict[str, Any]) -> str:
        candidates = chunk.get("candidates") or [{}]
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))

    def generate(self, system: str, prompt: str) -> LLMResult:
        usage = Usage(provider="gemini", model=self.model)
        url = f"/models/{self.model}:generateContent"
        for attempt in range(self.max_retries + 1):
            response = self.client.post(url, json=self._body(system, prompt))
            if response.status_code in RETRYABLE and attempt < self.max_retries:
                wait = 2 ** (attempt + 1)  # 2s, 4s, 8s: exponential backoff
                logger.warning("gemini %s, retrying in %ss", response.status_code, wait)
                time.sleep(wait)
                continue
            if response.is_error:
                raise LLMError(f"gemini {response.status_code}: {response.text[:300]}")
            data = response.json()
            self._usage(usage, data.get("usageMetadata", {}))
            usage.stop()
            return LLMResult(text=self._text(data), usage=usage)
        raise LLMError("gemini: retries exhausted")

    def stream(self, system: str, prompt: str, usage: Usage) -> Iterator[str]:
        usage.provider, usage.model = "gemini", self.model
        url = f"/models/{self.model}:streamGenerateContent"
        # alt=sse -> Server-Sent Events: lines like `data: {...json...}`
        with self.client.stream(
            "POST", url, params={"alt": "sse"}, json=self._body(system, prompt)
        ) as response:
            if response.is_error:
                response.read()
                raise LLMError(f"gemini {response.status_code}: {response.text[:300]}")
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                chunk = json.loads(line[len("data: ") :])
                if "usageMetadata" in chunk:
                    self._usage(usage, chunk["usageMetadata"])
                text = self._text(chunk)
                if text:
                    yield text
        usage.stop()
