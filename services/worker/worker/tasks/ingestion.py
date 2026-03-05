"""
Celery task: ``ingest_document``

Receives a JSON payload ``{"document_id": "<uuid>"}`` and runs the full
ingestion pipeline (chunk → embed → upsert to Weaviate → persist to PG).

Retry strategy
--------------
* Up to 3 automatic retries on *any* exception.
* Exponential back-off: 30 s, 60 s, 120 s.
* ``acks_late=True`` (inherited from Celery app config) — the task message
  stays in the broker queue until the task returns successfully, so a crashed
  worker does not lose work.
* On permanent failure (max retries exhausted) Celery marks the task FAILURE.
  The ingestion service also marks the job and document FAILED in PostgreSQL so
  the API can surface a meaningful error to the caller.

The task is intentionally *thin*: it delegates all business logic to
``SyncIngestionService``.
"""
from __future__ import annotations

import logging

from celery.exceptions import MaxRetriesExceededError

from worker.application.ingestion_service import DocumentNotFoundError, SyncIngestionService
from worker.celery_app import celery_app
from worker.core.config import WorkerSettings

logger = logging.getLogger(__name__)

# Retry back-off delays (seconds) indexed by retry count (0-based)
_RETRY_DELAYS = [30, 60, 120]


@celery_app.task(
    bind=True,
    name="worker.tasks.ingestion.ingest_document",
    queue="ingestion",
    max_retries=3,
    acks_late=True,
    # Raise MaxRetriesExceededError instead of silently ignoring it
    throws=(DocumentNotFoundError,),
)
def ingest_document(self, payload: dict) -> dict:  # type: ignore[override]
    """
    Ingest a document that has already been stored in PostgreSQL.

    Parameters
    ----------
    payload : dict
        Must contain ``{"document_id": "<uuid4_hex>"}`` and nothing else.
        Keeping the payload minimal (<100 bytes) means Redis memory is never
        the ingestion bottleneck, even with thousands of queued tasks.

    Returns
    -------
    dict
        ``{"document_id": "...", "chunks_created": N}``
        Stored in the Celery result backend (Redis) for 24 hours.
    """
    document_id: str = payload["document_id"]
    task_id: str = self.request.id

    logger.info(
        "ingest_document received",
        extra={"document_id": document_id, "task_id": task_id},
    )

    try:
        settings = WorkerSettings()
        service = SyncIngestionService(settings)
        chunks_created = service.run(task_id=task_id, document_id=document_id)

    except DocumentNotFoundError:
        # Document was deleted between enqueue and execution — do not retry.
        logger.error(
            "Document not found — skipping retries",
            extra={"document_id": document_id, "task_id": task_id},
        )
        raise  # Celery marks task FAILURE immediately

    except MaxRetriesExceededError:
        logger.error(
            "Max retries exceeded for ingest_document",
            extra={"document_id": document_id, "task_id": task_id},
        )
        raise

    except Exception as exc:
        retry_count = self.request.retries
        delay = _RETRY_DELAYS[min(retry_count, len(_RETRY_DELAYS) - 1)]

        logger.warning(
            "Ingestion error — scheduling retry %d/%d in %ds: %s",
            retry_count + 1,
            self.max_retries,
            delay,
            exc,
            extra={"document_id": document_id, "task_id": task_id},
        )

        raise self.retry(exc=exc, countdown=delay)

    logger.info(
        "ingest_document succeeded",
        extra={
            "document_id": document_id,
            "task_id": task_id,
            "chunks_created": chunks_created,
        },
    )
    return {"document_id": document_id, "chunks_created": chunks_created}
