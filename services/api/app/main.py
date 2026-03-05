"""
FastAPI application factory with lifespan resource management.

Lifespan hook (startup / shutdown)
-----------------------------------
All expensive, shared resources are created once at startup and disposed at
shutdown:

* ``AsyncEngine`` + ``async_sessionmaker`` — shared across all requests
* ``AsyncOllamaEmbedder`` — single HTTPX connection pool
* ``AsyncOllamaLLMClient`` — single HTTPX connection pool
* ``AsyncWeaviateVectorStore`` — single Weaviate gRPC channel
* ``Celery`` app — lightweight broker client (no worker threads)

Resources are attached to ``app.state`` so that FastAPI's ``Depends()`` system
can inject them into route handlers without global variables.

Error responses
---------------
All unhandled application errors are caught by the registered exception handlers
and returned as generic JSON ``{"detail": "..."}`` — no stack traces are ever
sent to the client.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from celery import Celery
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.middleware import RequestContextMiddleware
from app.api.routes import documents, jobs, search
from app.application.errors import AppError, NotFoundError, PromptInjectionError, ValidationError
from app.application.reranking import CrossEncoderReranker
from app.core.config import Settings
from app.core.logging import configure_logging
from app.infrastructure.database import build_async_engine, build_session_factory
from app.infrastructure.embedding_ollama import AsyncOllamaEmbedder
from app.infrastructure.llm_ollama import AsyncOllamaLLMClient
from app.infrastructure.vector_store_weaviate import AsyncWeaviateVectorStore
from app.schemas.responses import HealthResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create resources.  Shutdown: dispose them."""
    settings = Settings()
    configure_logging(settings.log_level)

    logger.info("API service starting up")

    # ── PostgreSQL ─────────────────────────────────────────────────────────────
    engine = build_async_engine(settings)
    session_factory = build_session_factory(engine)
    app.state.session_factory = session_factory

    # ── Ollama ─────────────────────────────────────────────────────────────────
    embedder = AsyncOllamaEmbedder(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embed_model,
    )
    llm = AsyncOllamaLLMClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_llm_model,
    )
    app.state.embedder = embedder
    app.state.llm = llm

    # ── Weaviate ───────────────────────────────────────────────────────────────
    vector_store = AsyncWeaviateVectorStore(weaviate_url=settings.weaviate_url)
    app.state.vector_store = vector_store

    # ── Celery (broker client only — no worker threads here) ──────────────────
    celery_app = Celery(broker=settings.celery_broker_url)
    celery_app.conf.task_serializer = "json"
    celery_app.conf.accept_content = ["json"]
    app.state.celery_app = celery_app

    # ── Cross-encoder reranker (optional) ─────────────────────────────────────
    # Enabled when CROSS_ENCODER_MODEL is set in .env.
    # Falls back to BM25 hybrid if not configured.
    app.state.cross_encoder = None
    if settings.cross_encoder_model:
        try:
            app.state.cross_encoder = CrossEncoderReranker(settings.cross_encoder_model)
            logger.info("Cross-encoder reranker enabled: %s", settings.cross_encoder_model)
        except Exception:
            logger.exception(
                "Failed to load cross-encoder '%s' — falling back to BM25 hybrid",
                settings.cross_encoder_model,
            )

    logger.info("All resources initialised — API ready")

    yield  # Application runs here

    # ── Shutdown ───────────────────────────────────────────────────────────────
    logger.info("API service shutting down — disposing resources")
    await embedder.aclose()
    await llm.aclose()
    vector_store.close()
    await engine.dispose()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Legal RAG API",
        version="1.0.0",
        description=(
            "Hierarchical French legal document retrieval and question-answering service. "
            "Documents are ingested asynchronously via Celery workers."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        # Never expose internal error details via the default FastAPI exception handler
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── Middleware ─────────────────────────────────────────────────────────────
    app.add_middleware(RequestContextMiddleware)

    # ── Exception handlers ────────────────────────────────────────────────────
    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc)},
        )

    @app.exception_handler(PromptInjectionError)
    async def injection_error_handler(request: Request, exc: PromptInjectionError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc)},
        )

    @app.exception_handler(NotFoundError)
    async def not_found_handler(request: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"detail": str(exc)},
        )

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        logger.exception("Unhandled application error", extra={"error_type": type(exc).__name__})
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal error occurred."},
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception", extra={"error_type": type(exc).__name__})
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal error occurred."},
        )

    # ── Routes ────────────────────────────────────────────────────────────────
    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", service="api")

    app.include_router(documents.router, prefix="/v1")
    app.include_router(jobs.router,      prefix="/v1")
    app.include_router(search.router,    prefix="/v1")

    return app


app = create_app()
