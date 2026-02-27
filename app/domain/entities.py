from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Document:
    id: str
    title: str | None
    source: str | None
    text: str
    created_at: datetime


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    text: str
    index: int
    hierarchy_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
