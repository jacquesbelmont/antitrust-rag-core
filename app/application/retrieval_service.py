from __future__ import annotations

from app.application.errors import StorageError
from app.domain.entities import RetrievedChunk
from app.domain.ports import Embedder, VectorStore


class RetrievalService:
    def __init__(self, *, vector_store: VectorStore, embedder: Embedder) -> None:
        self._vector_store = vector_store
        self._embedder = embedder

    async def retrieve(self, *, query: str, top_k: int) -> list[RetrievedChunk]:
        try:
            qv = await self._embedder.embed_text(text=query)
            return await self._vector_store.search(query_vector=qv, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            raise StorageError("retrieval_failed") from exc
