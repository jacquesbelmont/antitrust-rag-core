from __future__ import annotations

from app.application.chunking.legal_chunker import split_legal_text_hierarchical
from app.application.errors import StorageError
from app.core.logging import get_logger
from app.domain.entities import Chunk
from app.domain.ports import ChunkRepository, Clock, DocumentRepository, Embedder, IdGenerator, VectorStore

logger = get_logger(__name__)


class IngestionService:
    def __init__(
        self,
        *,
        documents: DocumentRepository,
        chunks_repo: ChunkRepository,
        vector_store: VectorStore,
        embedder: Embedder,
        ids: IdGenerator,
        clock: Clock,
    ) -> None:
        self._documents = documents
        self._chunks_repo = chunks_repo
        self._vector_store = vector_store
        self._embedder = embedder
        self._ids = ids
        self._clock = clock

    async def ingest(self, *, title: str | None, source: str | None, text: str) -> tuple[str, int]:
        document_id = self._ids.new_id()
        await self._documents.add_document(document_id=document_id, title=title, source=source, text=text)

        drafts = split_legal_text_hierarchical(text=text)
        vectors = await self._embedder.embed_texts(texts=[d.text for d in drafts])

        chunks: list[Chunk] = []
        for idx, (draft, vec) in enumerate(zip(drafts, vectors, strict=True)):
            chunk_id = self._ids.new_id()
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    text=draft.text,
                    index=idx,
                    hierarchy_path=draft.hierarchy_path,
                    metadata={"vector": vec, "created_at": self._clock.now().isoformat()},
                )
            )

        try:
            await self._chunks_repo.add_chunks(chunks=chunks)
            await self._vector_store.upsert_chunks(chunks=chunks)
        except Exception as exc:  # noqa: BLE001
            raise StorageError("ingestion_failed") from exc

        logger.info(
            "document_ingested",
            extra={"event": "document_ingested", "document_id": document_id},
        )

        return document_id, len(chunks)
