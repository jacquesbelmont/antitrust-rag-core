"""
Pure domain entities — no framework dependencies, no I/O, fully immutable.

Design notes:
  - `frozen=True` prevents accidental mutation after construction.
  - `list[str]` is used for `hierarchy_path` (JSON-serialisable); the field
    is deliberately *not* a tuple so that downstream code can serialise it
    to/from JSONB without a custom encoder.
  - `metadata` is typed `dict[str, Any]` for extensibility. Callers must
    not mutate it after construction — treat it as logically immutable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Document:
    """A legal document as ingested by the system."""

    id: str
    title: str | None
    source: str | None
    # Full raw text — used by the worker for chunking; stored in PostgreSQL.
    text: str
    created_at: datetime


@dataclass(frozen=True)
class Chunk:
    """A contiguous, hierarchy-aware excerpt of a Document."""

    id: str
    document_id: str
    text: str
    # Sequential position within the parent document (0-based).
    index: int
    # Ordered list of ancestor headings, e.g.
    # ["ARRÊT du 15 jan 2024", "TITRE I", "SECTION 2"]
    hierarchy_path: list[str]
    # Extensible bag for retrieval-time metadata (scores, page numbers, …).
    metadata: dict[str, Any] = field(default_factory=dict)
    # Weaviate object UUID assigned after upsert; None until the vector is stored.
    vector_id: str | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    """A Chunk returned by a vector-similarity search, with its relevance score."""

    chunk: Chunk
    # Normalised cosine similarity in [0, 1] — higher is more relevant.
    score: float
