"""One interface for every LLM provider: the start of the reusable `paperpilot-core` library.

The rest of the app only knows `LLM`. Switching Gemini <-> Ollama <-> OpenAI/Claude is a config
change, and tests use a fake. Every call reports token usage and latency, which is what the
observability dashboard (later) is built on.
"""

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Usage:
    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    _started: float = field(default_factory=time.perf_counter, repr=False)

    def stop(self) -> None:
        self.latency_ms = int((time.perf_counter() - self._started) * 1000)


@dataclass
class LLMResult:
    text: str
    usage: Usage


class LLMError(RuntimeError):
    """Raised when the provider fails after retries. Routes turn it into HTTP 502/503."""


class LLM(Protocol):
    def generate(self, system: str, prompt: str) -> LLMResult: ...

    def stream(self, system: str, prompt: str, usage: Usage) -> Iterator[str]:
        """Yield text pieces as they arrive. Fills `usage` when the stream ends."""
        ...
