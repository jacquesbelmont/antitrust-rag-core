from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Protocol

from app.domain.entities import Chunk, RetrievedChunk


class VectorStore(ABC):
    @abstractmethod
    async def upsert_chunks(self, *, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def search(self, *, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        raise NotImplementedError


class Embedder(Protocol):
    async def embed_text(self, *, text: str) -> list[float]:
        ...

    async def embed_texts(self, *, texts: list[str]) -> list[list[float]]:
        ...


class LLMClient(Protocol):
    async def generate(self, *, prompt: str) -> str:
        ...


class IdGenerator(Protocol):
    def new_id(self) -> str:
        ...


class Clock(Protocol):
    def now(self):
        ...


class DocumentRepository(Protocol):
    async def add_document(self, *, document_id: str, title: str | None, source: str | None, text: str) -> None:
        ...

    async def get_document_text(self, *, document_id: str) -> str | None:
        ...


class ChunkRepository(Protocol):
    async def add_chunks(self, *, chunks: list[Chunk]) -> None:
        ...

    async def list_chunks(self, *, document_id: str) -> list[Chunk]:
        ...
