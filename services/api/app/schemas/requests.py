"""HTTP request body schemas (Pydantic V2)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentUploadRequest(BaseModel):
    """
    Payload for ``POST /v1/documents/upload``.

    The full document text must be extracted *before* calling this endpoint
    (e.g. from a PDF via PyMuPDF on the client or a separate extraction step).
    The worker only performs chunking + embedding — it does not re-read the
    original file.
    """

    title: str | None = Field(
        default=None,
        max_length=512,
        description="Human-readable document title (optional).",
    )
    source: str | None = Field(
        default=None,
        max_length=1024,
        description="Source identifier — file path, URL, or reference (optional).",
    )
    text: str = Field(
        min_length=1,
        max_length=2_000_000,  # ~2 MB of text
        description="Full document text (UTF-8).",
    )


class SearchRequest(BaseModel):
    """Payload for ``POST /v1/search/``."""

    query: str = Field(
        min_length=1,
        max_length=1_000,
        description="Legal question to search for.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of context chunks to retrieve.",
    )
