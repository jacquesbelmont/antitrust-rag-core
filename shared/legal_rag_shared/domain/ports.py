"""
Port interfaces — the hexagonal boundary between domain and infrastructure.

All network-bound operations are `async`. Pure in-process helpers (id
generation, clock) remain synchronous so that domain logic stays free of
`await` boilerplate.

Usage:
  - API service:    implement `DocumentRepository`, `ChunkRepository`,
                    `VectorStore`, `Embedder`, `LLMClient`
  - Worker service: uses its own sync repositories internally;
                    still implements `VectorStore` and `Embedder`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Protocol, runtime_checkable

from legal_rag_shared.domain.entities import Chunk, Document, RetrievedChunk


# ── Embedding ─────────────────────────────────────────────────────────────────

@runtime_checkable
class Embedder(Protocol):
    """Converts text into dense vector representations."""

    async def embed_text(self, text: str) -> list[float]:
        """Return a single embedding vector for *text*."""
        ...

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return a list of embedding vectors, one per element of *texts*.

        Implementations should process texts in a single batched request
        where the underlying model supports it.
        """
        ...


# ── LLM ───────────────────────────────────────────────────────────────────────

@runtime_checkable
class LLMClient(Protocol):
    """Generates a natural-language answer from a prompt.

    The optional *system* parameter carries the instruction/persona text and
    should be kept separate from the user-visible *prompt*.  Models served via
    a chat API (e.g. Ollama /api/chat, OpenAI) treat the system message at the
    attention level, leading to significantly better instruction following than
    prepending the system text to the user prompt.
    """

    async def generate(self, prompt: str, *, system: str = "") -> str:
        """Return the model's text completion for *prompt*.

        Parameters
        ----------
        prompt:
            The user-visible query / context string.
        system:
            Optional system-level instruction (persona, constraints).  When the
            underlying model supports a chat API, this is sent as the ``system``
            role; otherwise it is prepended to *prompt*.
        """
        ...


# ── Vector store ──────────────────────────────────────────────────────────────

class VectorStore(ABC):
    """Persist and search document chunk embeddings.

    Separated as an ABC (not Protocol) because concrete implementations
    carry non-trivial lifecycle state (connections, indices).
    """

    @abstractmethod
    async def upsert_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> list[str]:
        """Store *chunks* alongside their pre-computed *vectors*.

        Returns a list of opaque `vector_id` strings (one per chunk) that
        the caller can persist for later deletion or deduplication.

        Raises:
            ValueError: if `len(chunks) != len(vectors)`.
        """
        ...

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Return the *top_k* most similar chunks to *query_vector*."""
        ...

    @abstractmethod
    async def delete_by_document_id(self, document_id: str) -> int:
        """Delete all vectors belonging to *document_id*.

        Called before re-ingesting a document to prevent stale chunks from
        polluting search results (document versioning).

        Returns
        -------
        int
            Number of vectors deleted.
        """
        ...


# ── Repositories ──────────────────────────────────────────────────────────────

@runtime_checkable
class DocumentRepository(Protocol):
    """Persistence operations for Document aggregates."""

    async def save(self, document: Document) -> None:
        """Persist a new Document.  Raises if the ID already exists."""
        ...

    async def get_by_id(self, document_id: str) -> Document | None:
        """Return the Document with *document_id*, or None if not found."""
        ...


@runtime_checkable
class ChunkRepository(Protocol):
    """Persistence operations for Chunk aggregates."""

    async def save_batch(self, chunks: list[Chunk]) -> None:
        """Persist a batch of Chunks in a single operation."""
        ...

    async def list_by_document(self, document_id: str) -> list[Chunk]:
        """Return all chunks belonging to *document_id*, ordered by index."""
        ...


# ── Infrastructure helpers ────────────────────────────────────────────────────

@runtime_checkable
class IdGenerator(Protocol):
    """Produces unique, URL-safe string identifiers."""

    def new_id(self) -> str:
        """Return a new unique ID string."""
        ...


@runtime_checkable
class Clock(Protocol):
    """Provides the current UTC time.

    Injected as a dependency so that domain logic is trivially testable
    without mocking `datetime.utcnow()`.
    """

    def now(self) -> datetime:
        """Return the current UTC datetime (timezone-aware)."""
        ...
