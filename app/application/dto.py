from __future__ import annotations

from pydantic import BaseModel, Field


class IngestDocumentCommand(BaseModel):
    title: str | None = None
    source: str | None = None
    text: str = Field(min_length=1)


class IngestDocumentResult(BaseModel):
    document_id: str
    chunks_created: int


class SearchQuery(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class SearchResult(BaseModel):
    answer: str
    context: list[dict]
    evaluation: dict
