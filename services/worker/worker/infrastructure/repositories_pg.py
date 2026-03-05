"""
Synchronous PostgreSQL repositories for the Worker service.

These are plain classes with no async/await — Celery prefork workers run in
forked processes that do not have an event loop.

Responsibilities
----------------
SyncDocumentRepository  — load document content for chunking; update status.
SyncChunkRepository     — bulk-insert chunk records after the pipeline runs.
SyncJobRepository       — update the ingestion job audit record throughout
                          the task lifecycle (STARTED → SUCCESS / FAILURE).

Transaction boundary
--------------------
All methods flush (or execute) but do NOT commit.  The caller (Celery task)
owns the transaction via the `get_session` context manager in database.py.
This means the entire pipeline (update doc status → insert chunks → update
job) commits atomically in a single transaction.

Security
--------
All queries use SQLAlchemy ORM expressions — no raw SQL strings.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import delete as sa_delete, select, update
from sqlalchemy.orm import Session

from legal_rag_shared.db.models import (
    ChunkRecord,
    DocumentRecord,
    DocumentStatusEnum,
    IngestionJobRecord,
    IngestionJobStatusEnum,
)
from legal_rag_shared.domain.entities import Chunk, Document


# ── Document repository ───────────────────────────────────────────────────────

class SyncDocumentRepository:
    """Load and update Document records from the worker process."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_by_id(self, document_id: str) -> Document | None:
        """Load the full document including `raw_text` for the chunking pipeline.

        Returns None if the document does not exist (e.g. deleted between
        enqueueing and execution — the task should treat this as a no-op).
        """
        record = self._session.get(DocumentRecord, document_id)
        if record is None:
            return None
        return _record_to_document(record)

    def update_status(
        self,
        document_id: str,
        status: DocumentStatusEnum,
    ) -> None:
        """Flip the document's lifecycle status.

        Executed twice per successful ingestion:
          1. PENDING → PROCESSING  (task started)
          2. PROCESSING → COMPLETED (pipeline finished) or → FAILED
        """
        stmt = (
            update(DocumentRecord)
            .where(DocumentRecord.id == document_id)
            .values(status=status, updated_at=_utcnow())
        )
        self._session.execute(stmt)


# ── Chunk repository ──────────────────────────────────────────────────────────

class SyncChunkRepository:
    """Bulk-insert chunk records produced by the ingestion pipeline."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def delete_by_document_id(self, document_id: str) -> int:
        """Delete all chunk records for *document_id*.

        Called before re-ingesting to prevent duplicate rows (document
        versioning).  Returns the count of deleted rows.
        """
        result = self._session.execute(
            sa_delete(ChunkRecord).where(ChunkRecord.document_id == document_id)
        )
        return result.rowcount

    def save_batch(
        self,
        chunks: list[Chunk],
        vector_ids: dict[str, str],
    ) -> None:
        """Persist all *chunks* in a single flush.

        Args:
            chunks:     Domain Chunk entities from the chunking pipeline.
            vector_ids: Mapping of chunk.id → Weaviate object UUID, returned
                        by `VectorStore.upsert_chunks`.  Stored on ChunkRecord
                        so the system can later delete/update individual vectors.

        The bulk add_all approach avoids N individual INSERT round-trips.
        SQLAlchemy will batch these into a single `executemany` call.
        """
        records = [
            _chunk_to_record(chunk, vector_ids.get(chunk.id))
            for chunk in chunks
        ]
        self._session.add_all(records)
        self._session.flush()


# ── Job repository ────────────────────────────────────────────────────────────

class SyncJobRepository:
    """Write the ingestion job audit trail from inside the Celery task."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def mark_started(self, task_id: str) -> None:
        """Transition the job to STARTED and record the start timestamp."""
        stmt = (
            update(IngestionJobRecord)
            .where(IngestionJobRecord.id == task_id)
            .values(
                status=IngestionJobStatusEnum.STARTED,
                started_at=_utcnow(),
            )
        )
        self._session.execute(stmt)

    def mark_success(self, task_id: str, chunks_created: int) -> None:
        """Transition the job to SUCCESS and record completion metadata."""
        stmt = (
            update(IngestionJobRecord)
            .where(IngestionJobRecord.id == task_id)
            .values(
                status=IngestionJobStatusEnum.SUCCESS,
                chunks_created=chunks_created,
                completed_at=_utcnow(),
            )
        )
        self._session.execute(stmt)

    def mark_failure(self, task_id: str, error_message: str) -> None:
        """Transition the job to FAILURE.

        The *error_message* is a sanitised, single-line summary — never a
        raw Python traceback, which could leak internal implementation details.
        """
        # Truncate to 1000 chars so error_message never exceeds TEXT column
        # limits and stays readable in monitoring dashboards.
        safe_error = error_message[:1000]
        stmt = (
            update(IngestionJobRecord)
            .where(IngestionJobRecord.id == task_id)
            .values(
                status=IngestionJobStatusEnum.FAILURE,
                error_message=safe_error,
                completed_at=_utcnow(),
            )
        )
        self._session.execute(stmt)

    def mark_retry(self, task_id: str, error_message: str) -> None:
        """Transition the job to RETRY (Celery will re-enqueue the task)."""
        safe_error = error_message[:1000]
        stmt = (
            update(IngestionJobRecord)
            .where(IngestionJobRecord.id == task_id)
            .values(
                status=IngestionJobStatusEnum.RETRY,
                error_message=safe_error,
            )
        )
        self._session.execute(stmt)


# ── Private mapping helpers ───────────────────────────────────────────────────

def _record_to_document(record: DocumentRecord) -> Document:
    return Document(
        id=record.id,
        title=record.title,
        source=record.source,
        text=record.raw_text,
        created_at=record.created_at,
    )


def _chunk_to_record(chunk: Chunk, vector_id: str | None) -> ChunkRecord:
    return ChunkRecord(
        id=chunk.id,
        document_id=chunk.document_id,
        text=chunk.text,
        chunk_index=chunk.index,
        hierarchy_path=chunk.hierarchy_path,
        chunk_metadata=chunk.metadata,
        vector_id=vector_id,
    )


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)
