"""
Synchronous Weaviate vector store for use inside Celery worker tasks.

Weaviate Python client v4 exposes a synchronous API natively — no
``asyncio.to_thread`` wrapper needed, making it straightforward to call from
Celery's prefork pool.

Collection schema
-----------------
Name:       LegalChunk
Vectorizer: none  (vectors are pre-computed by OllamaEmbedder)
Distance:   cosine
Properties: chunk_id, document_id, text, hierarchy_path (TEXT_ARRAY), chunk_index

HNSW config
-----------
ef_construction=128, max_connections=64 — balanced for read-heavy legal search
workloads.  Tune upward for higher recall at the cost of more RAM.
"""
from __future__ import annotations

import logging
from urllib.parse import urlparse

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter

from legal_rag_shared.domain.entities import Chunk

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "LegalChunk"


class SyncWeaviateVectorStore:
    """Synchronous Weaviate client wrapper for chunk upsert operations."""

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
        self._ensure_collection()

    # ── Public API ─────────────────────────────────────────────────────────────

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> dict[str, str]:
        """
        Upsert *chunks* with their pre-computed *vectors* into Weaviate.

        Parameters
        ----------
        chunks:
            Domain Chunk objects.
        vectors:
            Embedding vectors, same length and order as *chunks*.

        Returns
        -------
        dict[str, str]
            Mapping ``chunk.id → weaviate_uuid`` for every inserted chunk.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and vectors ({len(vectors)}) lengths differ"
            )

        collection = self._client.collections.get(_COLLECTION_NAME)
        id_map: dict[str, str] = {}
        objects: list[DataObject] = []

        for chunk, vector in zip(chunks, vectors):
            weaviate_uuid = _chunk_id_to_uuid(chunk.id)
            objects.append(
                DataObject(
                    properties={
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "text": chunk.text,
                        "hierarchy_path": chunk.hierarchy_path,
                        "chunk_index": chunk.index,
                    },
                    vector=vector,
                    uuid=weaviate_uuid,
                )
            )
            id_map[chunk.id] = weaviate_uuid

        # Batch upsert with fixed-size batches (100 objects each)
        with collection.batch.fixed_size(batch_size=100) as batch:
            for obj in objects:
                batch.add_object(
                    properties=obj.properties,
                    vector=obj.vector,
                    uuid=obj.uuid,
                )

        logger.info("Upserted %d chunks to Weaviate", len(objects))
        return id_map

    def delete_by_document_id(self, document_id: str) -> int:
        """Delete all Weaviate objects belonging to *document_id*.

        Used for document versioning: called before re-ingesting a document
        so that stale chunks don't pollute search results.

        Returns
        -------
        int
            Number of objects deleted (0 if the document had no prior chunks).
        """
        collection = self._client.collections.get(_COLLECTION_NAME)
        result = collection.data.delete_many(
            where=Filter.by_property("document_id").equal(document_id),
        )
        deleted = result.successful if result else 0
        if deleted > 0:
            logger.info(
                "Deleted %d stale Weaviate chunks for document %s",
                deleted, document_id,
            )
        return deleted

    def close(self) -> None:
        """Close the Weaviate connection."""
        self._client.close()

    def __enter__(self) -> "SyncWeaviateVectorStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        """Create the LegalChunk collection if it does not already exist."""
        if self._client.collections.exists(_COLLECTION_NAME):
            return

        self._client.collections.create(
            name=_COLLECTION_NAME,
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                ef_construction=128,
                max_connections=64,
                distance_metric=weaviate.classes.config.VectorDistances.COSINE,
            ),
            properties=[
                Property(name="chunk_id",       data_type=DataType.TEXT),
                Property(name="document_id",    data_type=DataType.TEXT),
                Property(name="text",           data_type=DataType.TEXT),
                Property(name="hierarchy_path", data_type=DataType.TEXT_ARRAY),
                Property(name="chunk_index",    data_type=DataType.INT),
            ],
        )
        logger.info("Created Weaviate collection '%s'", _COLLECTION_NAME)


def _chunk_id_to_uuid(chunk_id: str) -> str:
    """
    Convert an arbitrary chunk ID string to a Weaviate-compatible UUID v4 hex.

    Weaviate requires UUIDs as primary keys.  Since our chunk IDs are already
    UUID4 strings (from Uuid4IdGenerator), we normalise to ensure no dashes are
    present in a format Weaviate dislikes.
    """
    # Ensure 8-4-4-4-12 hyphenated format expected by Weaviate
    raw = chunk_id.replace("-", "")
    if len(raw) != 32:  # noqa: PLR2004
        raise ValueError(f"chunk_id '{chunk_id}' is not a valid UUID4")
    return f"{raw[:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:]}"
