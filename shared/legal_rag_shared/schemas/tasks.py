"""
Pydantic V2 schemas for inter-service communication.

`IngestDocumentPayload` is serialised as the Celery task argument.
`JobStatusResponse` is the API contract for GET /v1/jobs/{task_id}.

Design decisions:
  - The Celery message carries only `document_id`.  The worker fetches the
    full document (including raw text / PDF bytes) from PostgreSQL.  This
    keeps Redis messages small and avoids duplicating large payloads in the
    broker.
  - `model_config = {"frozen": True}` enforces immutability after construction
    so that task payloads cannot be mutated in transit.
  - All string fields have explicit `max_length` guards to prevent abnormally
    large values from propagating into the broker.
"""
from __future__ import annotations

import enum

from pydantic import BaseModel, Field


# ── Status enumerations ───────────────────────────────────────────────────────

class DocumentStatus(str, enum.Enum):
    """Lifecycle state of a document stored in PostgreSQL."""
    PENDING    = "pending"
    PROCESSING = "processing"
    COMPLETED  = "completed"
    FAILED     = "failed"


class CeleryTaskStatus(str, enum.Enum):
    """Maps directly onto Celery's internal task states."""
    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY   = "RETRY"
    REVOKED = "REVOKED"


# ── Task payload (API → Worker via Redis) ─────────────────────────────────────

class IngestDocumentPayload(BaseModel):
    """Celery task argument — intentionally minimal.

    The API creates the DocumentRecord in PostgreSQL *before* enqueueing
    this task.  The worker uses `document_id` to load the full content.
    """

    model_config = {"frozen": True}

    # Pre-generated UUID; the API owns ID creation so the worker is idempotent.
    document_id: str = Field(
        ...,
        description="UUID of the document already persisted in PostgreSQL.",
    )


# ── Task result (Worker → Celery result backend) ──────────────────────────────

class IngestDocumentResult(BaseModel):
    """Stored in the Celery result backend on task SUCCESS."""

    model_config = {"frozen": True}

    document_id: str
    chunks_created: int


# ── API response schema ───────────────────────────────────────────────────────

class JobStatusResponse(BaseModel):
    """Response contract for GET /v1/jobs/{task_id}."""

    task_id: str
    status: CeleryTaskStatus
    # Present only when status == SUCCESS
    result: IngestDocumentResult | None = None
    # Present only when status == FAILURE
    error: str | None = None
