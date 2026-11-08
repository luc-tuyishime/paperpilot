"""Local LLM through Ollama's chat API. Free and offline, but slow on 8 GB of RAM:
use a small model like llama3.2:3b. Same interface as Gemini, so it's a config switch."""

import json
from collections.abc import Iterator
from typing import Any

import httpx

from app.llm.base import LLMError, LLMResult, Usage


class OllamaLLM:
    def __init__(
        self, base_url: str, model: str, temperature: float = 0.1, timeout: float = 300.0
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.client = httpx.Client(base_url=base_url, timeout=timeout)

    def _body(self, system: str, prompt: str, stream: bool) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": stream,
            "options": {"temperature": self.temperature},
        }

    @staticmethod
    def _usage(usage: Usage, data: dict[str, Any]) -> None:
        usage.input_tokens = data.get("prompt_eval_count", 0)
        usage.output_tokens = data.get("eval_count", 0)

    def generate(self, system: str, prompt: str) -> LLMResult:
        usage = Usage(provider="ollama", model=self.model)
        try:
            response = self.client.post("/api/chat", json=self._body(system, prompt, False))
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama: {exc}") from exc
        data = response.json()
        self._usage(usage, data)
        usage.stop()
        return LLMResult(text=data["message"]["content"], usage=usage)

    def stream(self, system: str, prompt: str, usage: Usage) -> Iterator[str]:
        usage.provider, usage.model = "ollama", self.model
        with self.client.stream(
            "POST", "/api/chat", json=self._body(system, prompt, True)
        ) as response:
            if response.is_error:
                response.read()
                raise LLMError(f"ollama {response.status_code}: {response.text[:300]}")
            for line in response.iter_lines():  # one JSON object per line
                if not line:
                    continue
                data = json.loads(line)
                if data.get("done"):
                    self._usage(usage, data)
                    break
                text = data.get("message", {}).get("content", "")
                if text:
                    yield text
        usage.stop()
