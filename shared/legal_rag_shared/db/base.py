"""
Single source of truth for the SQLAlchemy DeclarativeBase.

All ORM models (DocumentRecord, ChunkRecord, IngestionJobRecord) must
inherit from this `Base` so that:
  1. `Base.metadata` reflects the complete schema for Alembic autogenerate.
  2. Both the API (async) and Worker (sync) services share an identical
     schema definition — only their engine/session factories differ.
"""
from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
