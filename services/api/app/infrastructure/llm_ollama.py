"""
Async Ollama LLM client — uses the Chat API (/api/chat).

Key design choices
------------------
* **Chat API over Generate API**: ``/api/chat`` sends system and user messages
  as separate roles.  Models trained with RLHF / instruction-tuning (Llama 3,
  Mistral Instruct, etc.) attend to the system message at a different layer than
  user content, producing significantly better instruction following than
  prepending the system text to a single ``prompt`` string.

* ``temperature=0.0`` — deterministic output for legal QA (reproducibility).
* ``num_predict=1024`` — caps output tokens to keep latency bounded while still
  allowing full answers for multi-point legal questions.

Setup: ``ollama pull llama3.1`` (or any chat-capable model)
"""
from __future__ import annotations

import logging

import httpx

from legal_rag_shared.domain.ports import LLMClient

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 120.0


class AsyncOllamaLLMClient:
    """Async Ollama chat client using /api/chat."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )

    async def generate(self, prompt: str, *, system: str = "") -> str:
        """
        Send a chat request to Ollama.

        Parameters
        ----------
        prompt:
            User-visible query with context chunks embedded.
        system:
            System-level instruction (persona, grounding rules).  Sent as the
            ``system`` role message — kept separate from the user query so the
            model honours it at the attention level.
        """
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.post(
            "/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 1024,
                },
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    async def aclose(self) -> None:
        await self._client.aclose()


# Protocol conformance check
def _check() -> None:
    _: LLMClient = AsyncOllamaLLMClient.__new__(AsyncOllamaLLMClient)  # type: ignore[assignment]

_check()
