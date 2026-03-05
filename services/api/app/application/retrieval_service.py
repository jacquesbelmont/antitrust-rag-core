"""
Retrieval service — embeds the query and searches the vector store.
"""
from __future__ import annotations

import logging

from legal_rag_shared.domain.entities import RetrievedChunk
from legal_rag_shared.domain.ports import Embedder, VectorStore
from app.application.errors import StorageError

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(self, vector_store: VectorStore, embedder: Embedder) -> None:
        self._store = vector_store
        self._embedder = embedder

    async def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Embed *query* and return the top-*top_k* nearest chunks."""
        try:
            query_vector = await self._embedder.embed_text(query)
            results = await self._store.search(query_vector, top_k=top_k)
        except Exception as exc:
            logger.exception("Retrieval failed", extra={"error_type": type(exc).__name__})
            raise StorageError(f"Retrieval failed: {exc}") from exc

        logger.debug("Retrieved %d chunks for query", len(results))
        return results
