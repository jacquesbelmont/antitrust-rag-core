"""
Async Weaviate vector store for the API service (read path only).

The API service only *reads* from Weaviate (search).  Write operations
(upsert) happen exclusively in the Celery worker.  ``asyncio.to_thread``
wraps the sync Weaviate v4 SDK so that search calls are non-blocking in the
async request handler.

Collection schema must match the one created by the worker:
    Name:       LegalChunk
    Vectorizer: none
    Distance:   cosine
    Properties: chunk_id, document_id, text, hierarchy_path, chunk_index
"""
from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse

import weaviate
from weaviate.classes.query import MetadataQuery

from legal_rag_shared.domain.entities import Chunk, RetrievedChunk
from legal_rag_shared.domain.ports import VectorStore

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "LegalChunk"


class AsyncWeaviateVectorStore(VectorStore):
    """Async wrapper around the sync Weaviate v4 Python client."""

    def __init__(self, weaviate_url: str) -> None:
        parsed = urlparse(weaviate_url)
        host = parsed.hostname or "weaviate"
        port = parsed.port or 8080
        secure = parsed.scheme == "https"

        self._client = weaviate.connect_to_custom(
            http_host=host,
            http_port=port,
            http_secure=secure,
            grpc_host=host,
            grpc_port=50051,
            grpc_secure=secure,
        )

    # ── VectorStore ABC ────────────────────────────────────────────────────────

    async def upsert_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> list[str]:
        """Not used in the API service — implemented to satisfy the ABC."""
        raise NotImplementedError(
            "Chunk upsert must be performed by the worker service, not the API."
        )

    async def delete_by_document_id(self, document_id: str) -> int:
        """Not used in the API service — implemented to satisfy the ABC."""
        raise NotImplementedError(
            "Chunk deletion must be performed by the worker service, not the API."
        )

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Run ANN search in a thread pool (non-blocking)."""
        return await asyncio.to_thread(self._sync_search, query_vector, top_k)

    def _sync_search(
        self,
        query_vector: list[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        collection = self._client.collections.get(_COLLECTION_NAME)
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )

        results: list[RetrievedChunk] = []
        for obj in response.objects:
            props = obj.properties
            distance = obj.metadata.distance if obj.metadata else 1.0
            # Weaviate cosine distance ∈ [0, 2]; similarity = 1 - distance/2
            score = max(0.0, 1.0 - (distance or 0.0) / 2.0)

            chunk = Chunk(
                id=props.get("chunk_id", str(obj.uuid)),
                document_id=props.get("document_id", ""),
                text=props.get("text", ""),
                index=int(props.get("chunk_index", 0)),
                hierarchy_path=list(props.get("hierarchy_path") or []),
                metadata={},
            )
            results.append(RetrievedChunk(chunk=chunk, score=score))

        return results

    def close(self) -> None:
        self._client.close()
