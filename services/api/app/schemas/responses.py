"""HTTP response schemas (Pydantic V2)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from legal_rag_shared.schemas.tasks import CeleryTaskStatus, IngestDocumentResult


class DocumentUploadResponse(BaseModel):
    """Response for ``POST /v1/documents/upload``."""

    task_id: str
    document_id: str
    status: str = "queued"


class JobStatusResponse(BaseModel):
    """Response for ``GET /v1/jobs/{task_id}``."""

    task_id: str
    status: CeleryTaskStatus
    result: IngestDocumentResult | None = None
    error: str | None = None


class SearchResponse(BaseModel):
    """Response for ``POST /v1/search/``."""

    answer: str
    context: list[dict[str, Any]]
    evaluation: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    service: str
