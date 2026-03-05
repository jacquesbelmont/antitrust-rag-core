"""
Synchronous Ollama embedder for use inside Celery worker tasks.

Uses ``httpx.Client`` (sync) because Celery's prefork pool does not support
running an asyncio event loop per task.  A single client instance is intended
to be reused across the lifetime of a single task execution — not shared
between tasks, because the prefork pool gives each worker its own process.

Embedding model: nomic-embed-text (768-dim, multilingual, ~50 ms/chunk on CPU)
Setup:  ``ollama pull nomic-embed-text``
"""
from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

# Timeout for a single embedding request (seconds).  Long enough for the first
# cold-start of the model, short enough to surface network issues quickly.
_DEFAULT_TIMEOUT_S = 120.0


class OllamaEmbedder:
    """Sync Ollama embedding client (nomic-embed-text or any local model)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self._model = model
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )

    def embed_text(self, text: str) -> list[float]:
        """Embed a single *text* string, returning a float vector."""
        response = self._client.post(
            "/api/embeddings",
            json={"model": self._model, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_texts(self, texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
        """
        Embed *texts* using Ollama's /api/embed batch endpoint.

        Processes texts in sub-batches of *batch_size* to control memory
        pressure on the Ollama server.  Using the batch endpoint instead of
        per-chunk /api/embeddings calls reduces HTTP round-trips from N to
        ceil(N / batch_size) — typically 10x faster for large documents.

        Requires Ollama >= 0.1.31.

        Parameters
        ----------
        texts:
            List of strings to embed (order is preserved).
        batch_size:
            Texts per HTTP request.  32 is safe for most hardware; reduce if
            you see OOM errors on the Ollama host.
        """
        if not texts:
            return []

        vectors: list[list[float]] = []
        total = len(texts)

        for batch_start in range(0, total, batch_size):
            batch = texts[batch_start: batch_start + batch_size]
            response = self._client.post(
                "/api/embed",
                json={"model": self._model, "input": batch},
            )
            response.raise_for_status()
            data = response.json()

            # /api/embed returns {"embeddings": [[...], ...]}
            if "embeddings" not in data:
                raise RuntimeError(
                    f"Ollama /api/embed returned unexpected shape: {list(data.keys())}. "
                    "Upgrade Ollama to >= 0.1.31 to use batch embedding."
                )

            vectors.extend(data["embeddings"])

            if total > batch_size:
                logger.debug(
                    "Embedded %d / %d chunks",
                    min(batch_start + batch_size, total),
                    total,
                )

        return vectors

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> "OllamaEmbedder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
