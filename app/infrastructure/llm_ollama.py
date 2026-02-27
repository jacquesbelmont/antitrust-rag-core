from __future__ import annotations

import httpx

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class OllamaLLMClient:
    """Local LLM via Ollama — zero token cost, no external API dependency.

    Trade-offs:
    - Requires `ollama serve` running locally (or in Docker).
    - Mistral 7B has strong French support — well-suited for antitrust court decisions.
    - temperature=0.0 → deterministic, mandatory for legal QA (no hallucination variance).
    - Falls back to an error string instead of raising, so the API stays alive during demos.

    Swap checklist for production:
    - Replace with Claude / GPT-4o by implementing the same `LLMClient` Protocol.
    - Add retry + exponential back-off for network errors.
    - Stream responses for latency-sensitive UIs.
    """

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self._model = model or settings.ollama_model
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")

    async def generate(self, *, prompt: str) -> str:
        url = f"{self._base_url}/api/generate"
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 512,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json().get("response", "").strip()
        except httpx.ConnectError:
            logger.warning(
                "ollama_unavailable",
                extra={"event": "ollama_unavailable", "url": url, "model": self._model},
            )
            return "[LLM unavailable — run: ollama serve && ollama pull " + self._model + "]"
        except Exception:
            logger.exception("ollama_generate_error", extra={"event": "ollama_generate_error"})
            raise
