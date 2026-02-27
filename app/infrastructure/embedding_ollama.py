from __future__ import annotations

import httpx

from app.core.config import settings
from app.core.logging import get_logger
from app.infrastructure.embedding import DeterministicHashEmbedder

logger = get_logger(__name__)


class OllamaEmbedder:
    """Embedding via Ollama's local embedding endpoint.

    Trade-offs:
    - nomic-embed-text: 768-dim, multilingual, strong on French legal text (~50 ms/chunk on CPU).
    - mxbai-embed-large: 1024-dim, higher recall at 3× the cost — switch for production.
    - Texts are embedded sequentially; for bulk ingestion (430K docs) replace with batched
      parallel calls or a sentence-transformers pipeline with GPU.
    - Falls back to the deterministic hash embedder when Ollama is unreachable, so tests
      and CI stay green without a running Ollama instance.

    Setup:
        ollama pull nomic-embed-text
    """

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self._model = model or settings.ollama_embed_model
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._fallback = DeterministicHashEmbedder()

    async def embed_text(self, *, text: str) -> list[float]:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": self._model, "prompt": text},
                )
                resp.raise_for_status()
                return resp.json()["embedding"]
        except Exception:
            logger.warning(
                "ollama_embed_fallback",
                extra={"event": "ollama_embed_fallback", "model": self._model},
            )
            return await self._fallback.embed_text(text=text)

    async def embed_texts(self, *, texts: list[str]) -> list[list[float]]:
        # Sequential for simplicity; parallelise with asyncio.gather for throughput.
        return [await self.embed_text(text=t) for t in texts]
