"""Pick the LLM from settings. The only place that knows which providers exist."""

from app.core.config import Settings
from app.llm.base import LLM, LLMError
from app.llm.gemini import GeminiLLM
from app.llm.ollama import OllamaLLM


def build_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "gemini":
        if settings.gemini_api_key is None:
            raise LLMError("PAPERPILOT_GEMINI_API_KEY is not set (or use LLM_PROVIDER=ollama)")
        return GeminiLLM(
            api_key=settings.gemini_api_key.get_secret_value(),
            model=settings.gemini_model,
            temperature=settings.llm_temperature,
        )
    return OllamaLLM(
        base_url=settings.ollama_url,
        model=settings.ollama_chat_model,
        temperature=settings.llm_temperature,
    )
