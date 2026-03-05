"""
Celery application factory for the ingestion worker.

Configuration notes
-------------------
* ``task_acks_late=True`` — the task message is ACKed *after* the task
  returns, not when it is received.  This ensures that if the worker process
  dies mid-task the broker re-queues the message for another worker.
* ``worker_prefetch_multiplier=1`` — each worker process fetches exactly one
  task at a time.  Combined with ``acks_late`` this prevents message loss when
  a worker is killed under load.
* ``task_track_started=True`` — Celery records a STARTED state in the result
  backend when a task begins, enabling the API to return "processing" rather
  than the generic "PENDING".
* ``max_tasks_per_child`` is set in the Dockerfile CMD (``--max-tasks-per-child=50``)
  rather than here so it can be tuned per deployment without rebuilding.
* The task route maps ``ingest_document`` to the ``ingestion`` queue so that
  other queues (e.g. a future ``search`` queue) can be added without changing
  this file.
"""
from __future__ import annotations

import logging

from celery import Celery
from celery.signals import setup_logging

from worker.core.config import WorkerSettings

logger = logging.getLogger(__name__)


def create_celery_app() -> Celery:
    settings = WorkerSettings()

    app = Celery("legal_rag_worker")

    app.conf.update(
        # ── Broker / backend ──────────────────────────────────────────────────
        broker_url=settings.celery_broker_url,
        result_backend=settings.celery_result_backend,
        # ── Serialisation ─────────────────────────────────────────────────────
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        # ── Reliability ───────────────────────────────────────────────────────
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        task_reject_on_worker_lost=True,
        # ── Observability ─────────────────────────────────────────────────────
        task_track_started=True,
        # ── Result TTL: keep results for 24 h then expire ─────────────────────
        result_expires=86_400,
        # ── Task routing ──────────────────────────────────────────────────────
        task_routes={
            "worker.tasks.ingestion.ingest_document": {"queue": "ingestion"},
        },
        # ── Task modules to import at worker startup ───────────────────────────
        include=["worker.tasks.ingestion"],
        # ── Timezone ──────────────────────────────────────────────────────────
        timezone="UTC",
        enable_utc=True,
    )

    return app


celery_app = create_celery_app()


# ── Preserve our structured JSON logging when Celery starts ───────────────────
@setup_logging.connect
def _configure_logging(**kwargs: object) -> None:
    """
    Prevent Celery from overwriting the root logger configuration set up by
    ``worker.core.logging``.  Called once at worker startup.
    """
    import logging as _logging
    from worker.core.logging import configure_logging

    configure_logging()
    logger.debug("Celery logging configured via worker.core.logging")
