"""
GET /v1/jobs/{task_id}
-----------------------
Returns the current status of an ingestion job by combining two sources of truth:

1. **Celery result backend (Redis)** — for real-time task state (PENDING, STARTED,
   SUCCESS, FAILURE, RETRY, REVOKED) and the task result payload.
2. **PostgreSQL IngestionJobRecord** — for persisted metadata (started_at,
   completed_at, chunks_created, error_message) that survives Redis TTL expiry.

Reconciliation strategy
-----------------------
* If the Celery result is PENDING but the DB record shows SUCCESS/FAILURE, the
  Redis result has expired → trust the DB.
* Otherwise trust Celery (it has higher resolution during the task lifetime).
* If neither source has the task_id, return 404.
"""
from __future__ import annotations

import logging

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_celery_app, get_job_repo
from app.infrastructure.repositories_pg import AsyncJobRepository
from app.schemas.responses import JobStatusResponse
from legal_rag_shared.schemas.tasks import CeleryTaskStatus, IngestDocumentResult

router = APIRouter(prefix="/jobs", tags=["jobs"])
logger = logging.getLogger(__name__)


@router.get(
    "/{task_id}",
    response_model=JobStatusResponse,
    summary="Poll ingestion job status",
)
async def get_job_status(
    task_id: str,
    job_repo: AsyncJobRepository = Depends(get_job_repo),
    celery=Depends(get_celery_app),
) -> JobStatusResponse:
    """
    Return the current status of a previously enqueued ingestion job.

    ``status`` follows Celery naming conventions so that callers can use a
    single polling loop regardless of whether the result is fresh in Redis or
    read from PostgreSQL.
    """
    # ── Step 1: Check PostgreSQL ───────────────────────────────────────────────
    job_record = await job_repo.get_by_id(task_id)

    # ── Step 2: Check Celery result backend ───────────────────────────────────
    async_result: AsyncResult = AsyncResult(task_id, app=celery)
    celery_state: str = async_result.state  # "PENDING", "STARTED", "SUCCESS", …

    # ── Step 3: 404 if unknown task ───────────────────────────────────────────
    if job_record is None and celery_state == "PENDING":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{task_id}' not found.",
        )

    # ── Step 4: Determine canonical status ────────────────────────────────────
    result: IngestDocumentResult | None = None
    error: str | None = None

    if celery_state == "SUCCESS":
        celery_result = async_result.result or {}
        result = IngestDocumentResult(
            document_id=celery_result.get("document_id", ""),
            chunks_created=celery_result.get("chunks_created", 0),
        )
        canonical_status = CeleryTaskStatus.SUCCESS

    elif celery_state in ("FAILURE", "REVOKED"):
        canonical_status = CeleryTaskStatus(celery_state)
        if job_record and job_record.error_message:
            error = job_record.error_message
        else:
            exc = async_result.result
            error = str(exc) if exc else "Unknown error"

    elif celery_state == "STARTED":
        canonical_status = CeleryTaskStatus.STARTED

    elif celery_state == "RETRY":
        canonical_status = CeleryTaskStatus.RETRY

    else:
        # PENDING — may be genuinely queued or Redis TTL expired
        if job_record is not None:
            # Trust the DB record
            db_status = job_record.status.value.upper()
            try:
                canonical_status = CeleryTaskStatus(db_status)
            except ValueError:
                canonical_status = CeleryTaskStatus.PENDING

            if job_record.chunks_created is not None and job_record.error_message is None:
                result = IngestDocumentResult(
                    document_id=str(job_record.document_id or ""),
                    chunks_created=job_record.chunks_created,
                )
            elif job_record.error_message:
                error = job_record.error_message
        else:
            canonical_status = CeleryTaskStatus.PENDING

    logger.debug(
        "Job status polled",
        extra={"task_id": task_id, "status": canonical_status.value},
    )

    return JobStatusResponse(
        task_id=task_id,
        status=canonical_status,
        result=result,
        error=error,
    )
