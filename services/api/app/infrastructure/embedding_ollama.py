"""
Async Ollama embedder for the API service.

Uses ``httpx.AsyncClient`` so embedding calls are non-blocking inside FastAPI
async request handlers.  A single shared client (created in lifespan) is used
across all requests for connection-pool reuse.

Model: nomic-embed-text (768-dim, multilingual)
Setup: ``ollama pull nomic-embed-text``
"""
from __future__ import annotations

import logging

import httpx

from legal_rag_shared.domain.ports import Embedder

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 60.0


class AsyncOllamaEmbedder:
    """Async Ollama embedding client."""

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

    async def embed_text(self, text: str) -> list[float]:
        response = await self._client.post(
            "/api/embeddings",
            json={"model": self._model, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # Sequential — Ollama handles one request at a time on the local GPU
        return [await self.embed_text(t) for t in texts]

    async def aclose(self) -> None:
        await self._client.aclose()


# Protocol conformance check
def _check() -> None:
    _: Embedder = AsyncOllamaEmbedder.__new__(AsyncOllamaEmbedder)  # type: ignore[assignment]

_check()
