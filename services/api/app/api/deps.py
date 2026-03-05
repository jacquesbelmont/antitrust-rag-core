"""
Dependency injection container for FastAPI.

All expensive objects (DB engines, HTTP clients, Weaviate connections) are
created once at application startup (in ``main.py``'s lifespan hook) and stored
in ``app.state``.  Route handlers receive them via ``Depends()`` — no global
singletons, no module-level side effects.

Pattern
-------
``get_settings``  → reads Settings once per process (cached via functools.lru_cache).
``get_*``         → retrieve objects from ``app.state`` (set during lifespan).
``get_rag_service`` / ``get_ingestion_*`` → assemble application services on-demand
                    (cheap, because their dependencies are already created).
"""
from __future__ import annotations

import functools

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.application.context_formatting import ContextFormatter
from app.application.evaluation import EvaluationService
from app.application.rag_service import RAGService
from app.application.retrieval_service import RetrievalService
from app.core.config import Settings
from app.infrastructure.repositories_pg import (
    AsyncDocumentRepository,
    AsyncJobRepository,
)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton (reads env vars once)."""
    return Settings()


# ── Raw infrastructure objects (from app.state) ───────────────────────────────

async def get_db_session(request: Request) -> AsyncSession:
    """Yield a transactional async DB session from the shared session factory."""
    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_embedder(request: Request):
    return request.app.state.embedder


async def get_llm(request: Request):
    return request.app.state.llm


async def get_vector_store(request: Request):
    return request.app.state.vector_store


async def get_celery_app(request: Request):
    return request.app.state.celery_app


async def get_cross_encoder(request: Request):
    """Return the CrossEncoderReranker if configured, else None (BM25 is used)."""
    return getattr(request.app.state, "cross_encoder", None)


# ── Repository factories ───────────────────────────────────────────────────────

async def get_document_repo(
    session: AsyncSession = Depends(get_db_session),
) -> AsyncDocumentRepository:
    return AsyncDocumentRepository(session)


async def get_job_repo(
    session: AsyncSession = Depends(get_db_session),
) -> AsyncJobRepository:
    return AsyncJobRepository(session)


# ── Application service factories ─────────────────────────────────────────────

async def get_rag_service(
    request: Request,
    settings: Settings = Depends(get_settings),
    embedder=Depends(get_embedder),
    vector_store=Depends(get_vector_store),
    llm=Depends(get_llm),
    cross_encoder=Depends(get_cross_encoder),
) -> RAGService:
    retrieval = RetrievalService(vector_store=vector_store, embedder=embedder)
    return RAGService(
        retrieval_service=retrieval,
        formatter=ContextFormatter(),
        llm=llm,
        evaluator=EvaluationService(),
        enable_reranking=settings.rag_enable_reranking,
        cross_encoder=cross_encoder,
    )
