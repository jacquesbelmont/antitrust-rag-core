"""
PostgreSQL repository implementations for the API service.

All methods are `async` and operate against a caller-provided `AsyncSession`.
They do NOT commit — transaction boundaries are owned by the FastAPI dependency
`get_async_session`, which commits on success and rolls back on exception.

Repositories implemented here
------------------------------
AsyncDocumentRepository   — implements `DocumentRepository` port.
AsyncJobRepository        — API-specific; no domain port (job management is
                            an infrastructure concern, not a domain concern).

Security
--------
* All queries use SQLAlchemy's ORM / Core expression layer — never raw SQL
  strings — which provides parameterised query protection against SQL injection.
* Error messages returned to callers never include raw database exceptions;
  those are caught and re-raised as domain errors by the service layer.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from legal_rag_shared.db.models import (
    ChunkRecord,
    DocumentRecord,
    DocumentStatusEnum,
    IngestionJobRecord,
    IngestionJobStatusEnum,
)
from legal_rag_shared.domain.entities import Chunk, Document
from legal_rag_shared.domain.ports import ChunkRepository, DocumentRepository


# ── Document repository ───────────────────────────────────────────────────────

class AsyncDocumentRepository:
    """Async PostgreSQL implementation of the `DocumentRepository` port."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # --- DocumentRepository protocol -----------------------------------------

    async def save(self, document: Document) -> None:
        """Persist a new Document with status=PENDING.

        The session is NOT committed here — the caller (or the FastAPI
        session dependency) is responsible for the commit.

        Raises:
            sqlalchemy.exc.IntegrityError: if a document with the same `id`
                already exists (duplicate ingestion request).
        """
        record = DocumentRecord(
            id=document.id,
            title=document.title,
            source=document.source,
            raw_text=document.text,
            char_count=len(document.text),
            status=DocumentStatusEnum.PENDING,
            created_at=document.created_at,
            updated_at=document.created_at,
        )
        self._session.add(record)
        # Flush to surface constraint violations *before* the route handler
        # returns a 201 to the client.
        await self._session.flush()

    async def get_by_id(self, document_id: str) -> Document | None:
        """Return the Document entity, or None if not found.

        Note: `raw_text` is included.  Callers that only need metadata
        should use `get_status` to avoid loading potentially large text.
        """
        stmt = select(DocumentRecord).where(DocumentRecord.id == document_id)
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()
        if record is None:
            return None
        return _record_to_document(record)

    # --- Extended operations (beyond the minimal port) -----------------------

    async def get_status(self, document_id: str) -> DocumentStatusEnum | None:
        """Lightweight metadata-only lookup — does NOT load `raw_text`."""
        stmt = select(DocumentRecord.status).where(
            DocumentRecord.id == document_id
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def update_status(
        self,
        document_id: str,
        status: DocumentStatusEnum,
    ) -> None:
        """Flip the document lifecycle status without loading the full record."""
        stmt = (
            update(DocumentRecord)
            .where(DocumentRecord.id == document_id)
            .values(status=status, updated_at=_utcnow())
        )
        await self._session.execute(stmt)


# ── Chunk repository ──────────────────────────────────────────────────────────

class AsyncChunkRepository:
    """Async PostgreSQL implementation of the `ChunkRepository` port."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # --- ChunkRepository protocol --------------------------------------------

    async def save_batch(self, chunks: list[Chunk]) -> None:
        """Persist all *chunks* in one flush operation."""
        records = [_chunk_to_record(c) for c in chunks]
        self._session.add_all(records)
        await self._session.flush()

    async def list_by_document(self, document_id: str) -> list[Chunk]:
        """Return all chunks for *document_id* ordered by `chunk_index`."""
        stmt = (
            select(ChunkRecord)
            .where(ChunkRecord.document_id == document_id)
            .order_by(ChunkRecord.chunk_index)
        )
        result = await self._session.execute(stmt)
        return [_record_to_chunk(r) for r in result.scalars().all()]


# ── Job repository ────────────────────────────────────────────────────────────

class AsyncJobRepository:
    """Read-only async repository for ingestion job status (API side).

    The worker owns writes to `ingestion_jobs`.  The API only:
      1. Creates the initial PENDING record at enqueue time.
      2. Reads status for GET /v1/jobs/{task_id}.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, task_id: str, document_id: str) -> None:
        """Insert a PENDING job record at task-enqueue time."""
        record = IngestionJobRecord(
            id=task_id,
            document_id=document_id,
            status=IngestionJobStatusEnum.PENDING,
            created_at=_utcnow(),
        )
        self._session.add(record)
        await self._session.flush()

    async def get_by_id(self, task_id: str) -> IngestionJobRecord | None:
        """Return the raw ORM record (status, timestamps, error) for *task_id*."""
        stmt = select(IngestionJobRecord).where(
            IngestionJobRecord.id == task_id
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()


# ── Private mapping helpers ───────────────────────────────────────────────────

def _record_to_document(record: DocumentRecord) -> Document:
    return Document(
        id=record.id,
        title=record.title,
        source=record.source,
        text=record.raw_text,
        created_at=record.created_at,
    )


def _chunk_to_record(chunk: Chunk) -> ChunkRecord:
    return ChunkRecord(
        id=chunk.id,
        document_id=chunk.document_id,
        text=chunk.text,
        chunk_index=chunk.index,
        hierarchy_path=chunk.hierarchy_path,
        chunk_metadata=chunk.metadata,
        vector_id=chunk.vector_id,
    )


def _record_to_chunk(record: ChunkRecord) -> Chunk:
    return Chunk(
        id=record.id,
        document_id=record.document_id,
        text=record.text,
        index=record.chunk_index,
        hierarchy_path=list(record.hierarchy_path),
        metadata=dict(record.chunk_metadata),
        vector_id=record.vector_id,
    )


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


# ── Type-checker protocol conformance assertions ──────────────────────────────
# These lines are never executed at runtime; they exist solely to fail loudly
# at import time if either class drifts out of sync with its port definition.

def _assert_protocols() -> None:
    _: DocumentRepository = AsyncDocumentRepository.__new__(AsyncDocumentRepository)
    __: ChunkRepository = AsyncChunkRepository.__new__(AsyncChunkRepository)
