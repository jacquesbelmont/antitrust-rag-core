"""
Alembic environment configuration.

This file is executed by the `alembic` CLI tool when running migrations.

Key design decisions
---------------------
* Database URL is read from `ALEMBIC_DATABASE_URL` env var — never hardcoded.
* All ORM models are imported (via `legal_rag_shared.db.models`) so that
  `target_metadata` reflects the complete schema for autogenerate.
* Both `run_migrations_online` (connected) and `run_migrations_offline`
  (SQL script generation) modes are supported.
"""
from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# ── Make sure the shared package is importable ────────────────────────────────
# Alembic runs from the repo root; the shared package must be on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHARED_DIR = _REPO_ROOT / "shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

# ── Import models to register them with Base.metadata ─────────────────────────
# The import of `models` is intentionally side-effectful: it registers all
# ORM classes with the DeclarativeBase so autogenerate can detect changes.
from legal_rag_shared.db.base import Base
from legal_rag_shared.db import models as _models  # noqa: F401

target_metadata = Base.metadata

# ── Alembic config ────────────────────────────────────────────────────────────
alembic_cfg = context.config

# Inject the database URL from the environment — fail fast if unset.
_db_url = os.environ.get("ALEMBIC_DATABASE_URL")
if not _db_url:
    raise RuntimeError(
        "ALEMBIC_DATABASE_URL environment variable is not set.\n"
        "Example: export ALEMBIC_DATABASE_URL="
        "postgresql+psycopg2://legalrag:password@localhost:5432/legalrag"
    )
alembic_cfg.set_main_option("sqlalchemy.url", _db_url)

# Set up logging from the alembic.ini [loggers] section.
if alembic_cfg.config_file_name is not None:
    fileConfig(alembic_cfg.config_file_name)


# ── Migration runners ─────────────────────────────────────────────────────────

def run_migrations_offline() -> None:
    """Generate a SQL script without connecting to the database.

    Useful for reviewing changes before applying, or for CI/CD pipelines
    that produce SQL artefacts for a DBA to execute.
    """
    context.configure(
        url=_db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live database connection."""
    connectable = engine_from_config(
        alembic_cfg.get_section(alembic_cfg.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # single-use connection; no pool for migrations
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
