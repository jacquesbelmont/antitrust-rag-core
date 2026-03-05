"""
POST /v1/search/
-----------------
Runs the full RAG pipeline:
  sanitize query → embed → ANN search → (optional) BM25 rerank → LLM → evaluate

Error handling
--------------
* ``PromptInjectionError`` / ``ValidationError`` → 422 Unprocessable Entity
* ``StorageError`` (Weaviate/Ollama) → 503 Service Unavailable
* Any other exception → 500 Internal Server Error (no stack trace to client)
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_rag_service
from app.application.errors import PromptInjectionError, StorageError, ValidationError
from app.application.rag_service import RAGService
from app.schemas.requests import SearchRequest
from app.schemas.responses import SearchResponse

router = APIRouter(prefix="/search", tags=["search"])
logger = logging.getLogger(__name__)


@router.post(
    "/",
    response_model=SearchResponse,
    summary="Query the legal document corpus",
    response_description="LLM-generated answer with grounding context and evaluation metrics.",
)
async def search(
    body: SearchRequest,
    rag: RAGService = Depends(get_rag_service),
) -> SearchResponse:
    """
    Submit a legal question and receive an answer grounded in ingested documents.

    The query is sanitized before use — queries that contain prompt injection
    patterns are rejected with HTTP 422.
    """
    try:
        answer, context, evaluation = await rag.answer(
            raw_query=body.query,
            top_k=body.top_k,
        )
    except PromptInjectionError as exc:
        logger.warning(
            "Prompt injection attempt rejected",
            extra={"error_type": "PromptInjectionError"},
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except StorageError as exc:
        logger.exception("RAG pipeline storage error", extra={"error_type": "StorageError"})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search service temporarily unavailable. Please retry.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error in RAG pipeline", extra={"error_type": type(exc).__name__})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred.",
        ) from exc

    return SearchResponse(answer=answer, context=context, evaluation=evaluation)
