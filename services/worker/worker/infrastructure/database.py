"""
Synchronous SQLAlchemy engine and session factory for the Worker service.

Driver: psycopg2  (Celery tasks run in forked processes — the async asyncpg
driver does not work reliably across fork() boundaries).

The worker uses a single engine per process, shared across all tasks handled
by that process.  Celery's `max_tasks_per_child=50` setting recycles processes
before connection pools can accumulate state or memory leaks from PDF parsing.

Connection URL is assembled from individual env vars — never a single
DATABASE_URL string — to prevent credential leakage in log lines.

Pool settings
-------------
* `pool_size=5`         — workers are CPU-bound (embedding), not I/O-bound;
                          a small pool is sufficient.
* `max_overflow=5`      — burst headroom for concurrent task retries.
* `pool_pre_ping=True`  — validates idle connections (tasks may run minutes
                          apart; idle connections time out server-side).
"""
from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from worker.core.config import WorkerSettings


def build_sync_engine(settings: WorkerSettings) -> Engine:
    """Construct a synchronous `Engine` from the worker settings."""
    url = (
        f"postgresql+psycopg2://"
        f"{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}"
        f"/{settings.postgres_db}"
    )
    return create_engine(
        url,
        echo=settings.db_echo,
        pool_size=5,
        max_overflow=5,
        pool_pre_ping=True,
    )


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a reusable session factory bound to *engine*."""
    return sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
    )


@contextmanager
def get_session(factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    """Context manager that yields a `Session`, commits on success, rolls back
    on any exception, and always closes.

    Usage (inside a Celery task):
        with get_session(session_factory) as db:
            repo = SyncDocumentRepository(db)
            doc = repo.get_by_id(document_id)
    """
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
