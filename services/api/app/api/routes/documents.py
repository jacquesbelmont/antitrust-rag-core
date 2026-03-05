"""
Document upload routes
----------------------

POST /v1/documents/upload
    Accepts pre-extracted text (JSON).  Use when the caller has already parsed
    the document (e.g. via their own pipeline or a client-side PDF extractor).

POST /v1/documents/upload-pdf
    Accepts a raw PDF file (multipart/form-data).  The API service extracts
    text with PyMuPDF synchronously before returning 202, so callers receive
    a clear 422 for unreadable PDFs instead of a silent worker failure.

Both routes follow the same flow after text extraction:
  1. Persist Document + IngestionJobRecord rows in PostgreSQL.
  2. Commit the transaction explicitly.
  3. Enqueue the Celery ``ingest_document`` task (document_id only).

The explicit commit-before-enqueue ordering is critical: if the task were sent
before the commit, a fast worker could query PostgreSQL before the row is
visible → 404 in the worker.

Response: 202 Accepted + ``{task_id, document_id, status: "queued"}``

Security
--------
* Pydantic / FastAPI validates lengths and types before any DB write.
* PDF size is capped at 50 MB; content-type is checked against application/pdf.
* Raw text stored as-is; no HTML rendering path that could lead to XSS.
* Stack traces are never forwarded to the client.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_celery_app, get_db_session, get_document_repo, get_job_repo
from app.infrastructure.clock import UtcClock
from app.infrastructure.id_generator import Uuid4IdGenerator
from app.infrastructure.pdf_parser import PDFParseError, extract_text
from app.infrastructure.repositories_pg import AsyncDocumentRepository, AsyncJobRepository
from app.schemas.requests import DocumentUploadRequest
from app.schemas.responses import DocumentUploadResponse
from legal_rag_shared.domain.entities import Document

_MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB — guard against memory exhaustion

router = APIRouter(prefix="/documents", tags=["documents"])

logger = logging.getLogger(__name__)
_id_gen = Uuid4IdGenerator()
_clock = UtcClock()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document for async ingestion",
    response_description="Ingestion task queued; poll /v1/jobs/{task_id} for status.",
)
async def upload_document(
    body: DocumentUploadRequest,
    db: AsyncSession = Depends(get_db_session),
    doc_repo: AsyncDocumentRepository = Depends(get_document_repo),
    job_repo: AsyncJobRepository = Depends(get_job_repo),
    celery=Depends(get_celery_app),
) -> DocumentUploadResponse:
    """
    Store document text in PostgreSQL and enqueue an ingestion task.

    The Celery task carries only ``document_id``; the worker loads the full text
    from PostgreSQL, keeping the broker (Redis) message tiny.

    Transaction ordering
    --------------------
    We commit **before** sending the Celery task to guarantee that the worker
    finds the document row when it queries PostgreSQL.  The second commit call
    from the ``get_db_session`` generator cleanup is a safe no-op on an already-
    committed session (SQLAlchemy is idempotent for empty commits).
    """
    document_id = _id_gen.new_id()
    task_id = _id_gen.new_id()

    # ── Persist document + job records ────────────────────────────────────────
    document = Document(
        id=document_id,
        title=body.title,
        source=body.source,
        text=body.text,
        created_at=_clock.now(),
    )

    try:
        await doc_repo.save(document)
        await job_repo.create(task_id=task_id, document_id=document_id)
    except Exception as exc:
        logger.exception(
            "Failed to persist document or job record",
            extra={"error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store document.",
        ) from exc

    # ── Commit BEFORE enqueueing — worker must see the committed row ──────────
    try:
        await db.commit()
    except Exception as exc:
        logger.exception("DB commit failed before task enqueue", extra={"document_id": document_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store document.",
        ) from exc

    # ── Enqueue the Celery task ───────────────────────────────────────────────
    try:
        celery.send_task(
            "worker.tasks.ingestion.ingest_document",
            args=[{"document_id": document_id}],
            task_id=task_id,
            queue="ingestion",
        )
    except Exception as exc:
        logger.exception(
            "Failed to enqueue Celery task — document stored but not indexed",
            extra={"document_id": document_id, "task_id": task_id},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document stored but failed to queue for indexing. Please retry.",
        ) from exc

    logger.info(
        "Document upload accepted",
        extra={
            "document_id": document_id,
            "task_id": task_id,
            "char_count": len(body.text),
        },
    )

    return DocumentUploadResponse(
        task_id=task_id,
        document_id=document_id,
        status="queued",
    )


@router.post(
    "/upload-pdf",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a PDF document for async ingestion",
    response_description="Ingestion task queued; poll /v1/jobs/{task_id} for status.",
)
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file (max 50 MB, must have a text layer)"),
    title: str | None = Form(default=None, max_length=512),
    source: str | None = Form(default=None, max_length=1024),
    db: AsyncSession = Depends(get_db_session),
    doc_repo: AsyncDocumentRepository = Depends(get_document_repo),
    job_repo: AsyncJobRepository = Depends(get_job_repo),
    celery=Depends(get_celery_app),
) -> DocumentUploadResponse:
    """
    Accept a raw PDF file, extract its text, and enqueue an ingestion task.

    Text extraction (PyMuPDF) runs synchronously before the 202 is returned so
    that callers receive an immediate 422 for corrupt/encrypted/image PDFs
    rather than a silent failure discovered later in the worker.

    File constraints
    ----------------
    * Content-Type must be ``application/pdf``.
    * Maximum file size: 50 MB.
    * The PDF must have a selectable text layer (scanned images require OCR
      preprocessing and should use the ``/upload`` endpoint after conversion).
    """
    # ── Validate content-type ─────────────────────────────────────────────────
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type '{file.content_type}'. Expected application/pdf.",
        )

    # ── Read file bytes (enforces size cap) ───────────────────────────────────
    content = await file.read()
    if len(content) > _MAX_PDF_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"PDF exceeds the 50 MB upload limit ({len(content) // (1024 * 1024)} MB received).",
        )
    if not content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )

    # ── Extract text (CPU-bound — run in thread to free the event loop) ───────
    try:
        text = await asyncio.to_thread(extract_text, content)
    except PDFParseError as exc:
        logger.warning(
            "PDF extraction failed",
            extra={"filename": file.filename, "error": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    # ── Resolve metadata (fallback to filename when title not provided) ────────
    resolved_title = title or file.filename or "Untitled"
    resolved_source = source or file.filename

    document_id = _id_gen.new_id()
    task_id = _id_gen.new_id()

    # ── Persist document + job records ────────────────────────────────────────
    document = Document(
        id=document_id,
        title=resolved_title,
        source=resolved_source,
        text=text,
        created_at=_clock.now(),
    )

    try:
        await doc_repo.save(document)
        await job_repo.create(task_id=task_id, document_id=document_id)
    except Exception as exc:
        logger.exception(
            "Failed to persist PDF document or job record",
            extra={"error_type": type(exc).__name__, "filename": file.filename},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store document.",
        ) from exc

    # ── Commit BEFORE enqueueing — worker must see the committed row ──────────
    try:
        await db.commit()
    except Exception as exc:
        logger.exception("DB commit failed before task enqueue", extra={"document_id": document_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store document.",
        ) from exc

    # ── Enqueue the Celery task ───────────────────────────────────────────────
    try:
        celery.send_task(
            "worker.tasks.ingestion.ingest_document",
            args=[{"document_id": document_id}],
            task_id=task_id,
            queue="ingestion",
        )
    except Exception as exc:
        logger.exception(
            "Failed to enqueue Celery task — PDF stored but not indexed",
            extra={"document_id": document_id, "task_id": task_id},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document stored but failed to queue for indexing. Please retry.",
        ) from exc

    logger.info(
        "PDF upload accepted",
        extra={
            "document_id": document_id,
            "task_id": task_id,
            "filename": file.filename,
            "char_count": len(text),
        },
    )

    return DocumentUploadResponse(
        task_id=task_id,
        document_id=document_id,
        status="queued",
    )
