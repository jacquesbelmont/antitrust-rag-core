"""
SQLAlchemy 2.x ORM models — the relational schema for the Legal RAG system.

Tables
------
documents       Metadata + raw text for every ingested document.
chunks          Hierarchy-aware text excerpts derived from a document.
ingestion_jobs  Audit log and status tracker for every Celery ingestion task.

Design notes
------------
* All PKs are VARCHAR(36) UUIDs to stay consistent with the existing
  domain-layer IdGenerator (which produces uuid4 strings).
* JSONB is used for `hierarchy_path` and `chunk_metadata` because PostgreSQL
  can index JSONB columns and the worker writes/reads these as Python lists
  and dicts respectively.
* `raw_text` on DocumentRecord uses PostgreSQL's TOAST mechanism for
  transparent compression — legal documents often compress 70–80 %.
* `updated_at` uses a server-side `onupdate` trigger rather than
  application-side logic to avoid clock-skew in multi-worker deployments.
* Relationships are defined for ORM-level convenience (e.g. eager loading
  in tests) but are not required for the primary read/write paths.
"""
from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy import Enum as SaEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from legal_rag_shared.db.base import Base


# ── Enumerations ──────────────────────────────────────────────────────────────

class DocumentStatusEnum(str, enum.Enum):
    """Lifecycle state of a document; written to the `documents` table."""
    PENDING    = "pending"
    PROCESSING = "processing"
    COMPLETED  = "completed"
    FAILED     = "failed"


class IngestionJobStatusEnum(str, enum.Enum):
    """Maps 1-to-1 with Celery task states; written to `ingestion_jobs`."""
    PENDING = "pending"
    STARTED = "started"
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY   = "retry"
    REVOKED = "revoked"


# ── ORM models ────────────────────────────────────────────────────────────────

class DocumentRecord(Base):
    """Represents a single legal document stored in the system."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Full document text — TOASTed automatically by PostgreSQL.
    # The worker reads this to run the chunking pipeline.
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)

    # Precomputed for fast size-based queries without loading raw_text.
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    status: Mapped[DocumentStatusEnum] = mapped_column(
        SaEnum(DocumentStatusEnum, name="document_status", create_type=True,
               values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=DocumentStatusEnum.PENDING,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationships (convenience; not used in hot paths)
    chunks: Mapped[list[ChunkRecord]] = relationship(
        "ChunkRecord",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    jobs: Mapped[list[IngestionJobRecord]] = relationship(
        "IngestionJobRecord",
        back_populates="document",
        lazy="selectin",
    )


class ChunkRecord(Base):
    """A single hierarchy-aware text excerpt produced by the chunking pipeline."""

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )

    text: Mapped[str] = mapped_column(Text, nullable=False)

    # Sequential position within the parent document (0-based).
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # Ordered list of ancestor headings, e.g.
    # ["ARRÊT du 15 jan 2024", "TITRE I", "SECTION 2"]
    hierarchy_path: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, default=list
    )

    # Extensible metadata bag (page numbers, confidence scores, …)
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict
    )

    # Weaviate object UUID — populated by the worker after a successful upsert.
    # NULL means the chunk has not yet been indexed in the vector store.
    vector_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    document: Mapped[DocumentRecord] = relationship(
        "DocumentRecord",
        back_populates="chunks",
    )

    __table_args__ = (
        # Most queries filter by document_id; this index is critical for
        # the worker's save_batch and the API's list_by_document paths.
        Index("ix_chunks_document_id", "document_id"),
    )


class IngestionJobRecord(Base):
    """Audit log entry for a single Celery ingestion task.

    The `id` is the Celery task UUID so that GET /v1/jobs/{task_id} can
    join on this table without an extra lookup layer.
    """

    __tablename__ = "ingestion_jobs"

    # Celery task UUID — owned by the API at task-enqueue time.
    id: Mapped[str] = mapped_column(String(36), primary_key=True)

    # Nullable: the document may be deleted after the job is recorded.
    document_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )

    status: Mapped[IngestionJobStatusEnum] = mapped_column(
        SaEnum(IngestionJobStatusEnum, name="ingestion_job_status", create_type=True,
               values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=IngestionJobStatusEnum.PENDING,
    )

    # Populated on SUCCESS; useful for monitoring ingestion throughput.
    chunks_created: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Sanitised error message (no stack traces) written on FAILURE.
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    document: Mapped[DocumentRecord | None] = relationship(
        "DocumentRecord",
        back_populates="jobs",
    )

    __table_args__ = (
        Index("ix_ingestion_jobs_document_id", "document_id"),
        # Supports monitoring queries like "show all FAILED jobs".
        Index("ix_ingestion_jobs_status", "status"),
    )
