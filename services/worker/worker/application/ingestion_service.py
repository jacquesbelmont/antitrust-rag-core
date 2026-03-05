"""
Synchronous ingestion pipeline — runs inside a Celery worker process.

Pipeline stages
---------------
1. **Load**    — Fetch DocumentRecord from PostgreSQL; mark job as STARTED.
2. **Version** — Delete any existing chunks for this document from Weaviate + PG.
3. **Chunk**   — Split full text with the hierarchical legal chunker.
4. **Embed**   — Call Ollama (sync HTTP) to obtain float vectors per chunk.
5. **Upsert**  — Batch-insert chunks + vectors into Weaviate (sync client).
6. **Persist** — Write ChunkRecord rows to PostgreSQL; mark job SUCCESS / FAILED.

Error handling
--------------
Any exception raised during stages 2–4 is caught, the job is marked FAILED in
PostgreSQL, the document status is set to FAILED, and the exception is
re-raised so that Celery's retry / FAILURE machinery sees it.

A separate ``get_session`` context manager is used for the initial load (stage 1)
and for the final persist (stage 5), so the DB connection is *not* held open
during the expensive embedding / Weaviate stages.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from legal_rag_shared.db.models import DocumentStatusEnum
from legal_rag_shared.domain.entities import Chunk
from worker.application.chunking.legal_chunker import split_legal_text_hierarchical
from worker.core.config import WorkerSettings
from worker.infrastructure.clock import UtcClock
from worker.infrastructure.database import build_session_factory, build_sync_engine, get_session
from worker.infrastructure.embedding_ollama import OllamaEmbedder
from worker.infrastructure.id_generator import Uuid4IdGenerator
from worker.infrastructure.repositories_pg import (
    SyncChunkRepository,
    SyncDocumentRepository,
    SyncJobRepository,
)
from worker.infrastructure.vector_store_weaviate import SyncWeaviateVectorStore

logger = logging.getLogger(__name__)

_clock = UtcClock()
_id_gen = Uuid4IdGenerator()


class DocumentNotFoundError(RuntimeError):
    """The document_id was not found in PostgreSQL."""


class SyncIngestionService:
    """
    Orchestrates the full ingestion pipeline for a single document.

    Instantiated once per Celery task execution so that each task gets its own
    connection pool and HTTP client — there is no shared state between tasks.
    """

    def __init__(self, settings: WorkerSettings) -> None:
        self._settings = settings
        engine = build_sync_engine(settings)
        self._session_factory = build_session_factory(engine)

    # ── Public interface ───────────────────────────────────────────────────────

    def run(self, *, task_id: str, document_id: str) -> int:
        """
        Execute the full pipeline for *document_id*.

        Parameters
        ----------
        task_id:
            Celery task UUID — used as IngestionJobRecord.id.
        document_id:
            UUID of the DocumentRecord to ingest.

        Returns
        -------
        int
            Number of chunks created.

        Raises
        ------
        DocumentNotFoundError
            If the document is not found in PostgreSQL.
        Any other exception
            Pipeline error (embedding failure, Weaviate timeout, etc.).
            Before re-raising, the job and document are marked FAILED in PG.
        """
        # ── Stage 1: Mark started, load document ──────────────────────────────
        document_text = self._load_and_mark_started(task_id, document_id)

        # ── Stages 2-5: CPU / network — run outside DB session ───────────────
        try:
            self._delete_existing_chunks(document_id)  # Stage 2: versioning
            chunks, vector_id_map = self._chunk_embed_upsert(document_id, document_text)
        except Exception as exc:
            self._mark_failed(task_id, document_id, str(exc))
            raise

        # ── Stage 5: Persist chunks + mark success ────────────────────────────
        self._persist_and_mark_success(task_id, document_id, chunks, vector_id_map)
        return len(chunks)

    # ── Private pipeline stages ────────────────────────────────────────────────

    def _load_and_mark_started(self, task_id: str, document_id: str) -> str:
        """Open a short DB session: mark job started, flip doc status → PROCESSING, load text."""
        with get_session(self._session_factory) as session:
            job_repo = SyncJobRepository(session)
            doc_repo = SyncDocumentRepository(session)

            job_repo.mark_started(task_id)
            doc_repo.update_status(document_id, DocumentStatusEnum.PROCESSING)

            document = doc_repo.get_by_id(document_id)
            if document is None:
                raise DocumentNotFoundError(
                    f"Document '{document_id}' not found — was it deleted before the task ran?"
                )

        logger.info(
            "Document loaded from PG",
            extra={"document_id": document_id, "task_id": task_id},
        )
        return document.text

    def _delete_existing_chunks(self, document_id: str) -> None:
        """Delete stale chunks for *document_id* from Weaviate and PostgreSQL.

        This implements document versioning: re-uploading a document replaces
        its chunks rather than duplicating them.  Both stores are cleaned before
        the new chunks are created, so a partial failure leaves the document
        in a degraded (zero chunks) but consistent state — not a duplicated one.
        """
        # Delete from Weaviate first (external, no transaction)
        with SyncWeaviateVectorStore(self._settings.weaviate_url) as vector_store:
            weaviate_deleted = vector_store.delete_by_document_id(document_id)

        # Delete from PostgreSQL (transactional)
        with get_session(self._session_factory) as session:
            pg_deleted = SyncChunkRepository(session).delete_by_document_id(document_id)

        if weaviate_deleted or pg_deleted:
            logger.info(
                "Versioning: removed %d Weaviate + %d PG chunks before re-ingestion",
                weaviate_deleted,
                pg_deleted,
                extra={"document_id": document_id},
            )

    def _chunk_embed_upsert(
        self,
        document_id: str,
        text: str,
    ) -> tuple[list[Chunk], dict[str, str]]:
        """
        Chunk → Embed → Upsert.  No DB connection held during this stage.

        Returns
        -------
        tuple[list[Chunk], dict[str, str]]
            (domain Chunk objects, {chunk.id → weaviate_uuid})
        """
        # Stage 2: Chunk
        drafts = split_legal_text_hierarchical(
            text,
            max_chunk_chars=self._settings.rag_max_chunk_chars,
            chunk_overlap_chars=self._settings.rag_chunk_overlap_chars,
        )
        logger.info("Text split into %d chunks", len(drafts))

        # Build domain Chunk objects (IDs assigned here)
        now: datetime = _clock.now()
        chunks: list[Chunk] = [
            Chunk(
                id=_id_gen.new_id(),
                document_id=document_id,
                text=draft.text,
                index=i,
                hierarchy_path=draft.hierarchy_path,
                metadata={"char_count": len(draft.text)},
            )
            for i, draft in enumerate(drafts)
        ]

        # Stage 3: Embed
        with OllamaEmbedder(
            base_url=self._settings.ollama_base_url,
            model=self._settings.ollama_embed_model,
        ) as embedder:
            texts = [c.text for c in chunks]
            vectors = embedder.embed_texts(texts)

        logger.info("Embedded %d chunks via Ollama", len(vectors))

        # Stage 4: Upsert to Weaviate
        with SyncWeaviateVectorStore(self._settings.weaviate_url) as vector_store:
            vector_id_map = vector_store.upsert_chunks(chunks, vectors)

        logger.info("Upserted %d chunks to Weaviate", len(chunks))
        return chunks, vector_id_map

    def _persist_and_mark_success(
        self,
        task_id: str,
        document_id: str,
        chunks: list[Chunk],
        vector_id_map: dict[str, str],
    ) -> None:
        """Open a final DB session: persist chunks, flip doc status → COMPLETED, mark job SUCCESS."""
        with get_session(self._session_factory) as session:
            chunk_repo = SyncChunkRepository(session)
            doc_repo = SyncDocumentRepository(session)
            job_repo = SyncJobRepository(session)

            chunk_repo.save_batch(chunks, vector_id_map)
            doc_repo.update_status(document_id, DocumentStatusEnum.COMPLETED)
            job_repo.mark_success(task_id, len(chunks))

        logger.info(
            "Ingestion completed",
            extra={
                "document_id": document_id,
                "task_id": task_id,
                "chunks_created": len(chunks),
            },
        )

    def _mark_failed(self, task_id: str, document_id: str, error_message: str) -> None:
        """Best-effort: mark job and document as FAILED.  Swallows any secondary DB error."""
        try:
            with get_session(self._session_factory) as session:
                SyncJobRepository(session).mark_failure(task_id, error_message)
                SyncDocumentRepository(session).update_status(
                    document_id, DocumentStatusEnum.FAILED
                )
        except Exception:
            logger.exception(
                "Failed to write FAILED status to DB — original error was: %s",
                error_message,
                extra={"document_id": document_id, "task_id": task_id},
            )
