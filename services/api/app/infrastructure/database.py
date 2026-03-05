"""
Async SQLAlchemy engine and session factory for the API service.

Driver: asyncpg  (PostgreSQL native async protocol — no thread pool overhead)

Usage
-----
    from app.infrastructure.database import get_async_session

    async def my_endpoint(session: AsyncSession = Depends(get_async_session)):
        ...

Connection URL is assembled from individual env vars (never a single
DATABASE_URL secret that could leak credentials in logs).

Connection pool notes
---------------------
* `pool_size=10`        — baseline connections kept open between requests.
* `max_overflow=20`     — burst headroom; total max = 30.
* `pool_pre_ping=True`  — validates idle connections before use (guards
                          against PostgreSQL's `idle_in_transaction_timeout`
                          closing connections under the API).
* `pool_timeout=30`     — raise after 30 s if no connection is available
                          (avoids indefinite queue buildup during load spikes).
"""
from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings


def build_async_engine(settings: Settings) -> AsyncEngine:
    """Construct an `AsyncEngine` from the application settings.

    The caller is responsible for disposing the engine on shutdown
    (i.e. inside a FastAPI lifespan context).
    """
    url = (
        f"postgresql+asyncpg://"
        f"{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}"
        f"/{settings.postgres_db}"
    )
    return create_async_engine(
        url,
        echo=settings.db_echo,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_timeout=30,
    )


def build_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Return a reusable session factory bound to *engine*.

    `autoflush=False`  — we flush explicitly before commits so that we
                         control exactly when SQL is emitted.
    `expire_on_commit=False` — keeps ORM objects usable after `session.commit()`
                               without triggering lazy-load (critical in async).
    """
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        autoflush=False,
        expire_on_commit=False,
    )


async def get_async_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a per-request `AsyncSession`.

    The session is committed on success and rolled back on any exception,
    then always closed.  Callers must NOT commit or roll back manually —
    transaction management is the responsibility of this generator.

    Example (route usage):
        async def route(
            db: Annotated[AsyncSession, Depends(get_db_session)],
        ) -> ...:
    """
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
