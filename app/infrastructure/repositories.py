from __future__ import annotations

from dataclasses import dataclass

from app.domain.entities import Chunk
from app.domain.ports import ChunkRepository, DocumentRepository


@dataclass
class InMemoryDocumentRepository(DocumentRepository):
    def __init__(self) -> None:
        self._docs: dict[str, str] = {}

    async def add_document(self, *, document_id: str, title: str | None, source: str | None, text: str) -> None:
        self._docs[document_id] = text

    async def get_document_text(self, *, document_id: str) -> str | None:
        return self._docs.get(document_id)


@dataclass
class InMemoryChunkRepository(ChunkRepository):
    def __init__(self) -> None:
        self._chunks_by_doc: dict[str, list[Chunk]] = {}

    async def add_chunks(self, *, chunks: list[Chunk]) -> None:
        for c in chunks:
            self._chunks_by_doc.setdefault(c.document_id, []).append(c)

    async def list_chunks(self, *, document_id: str) -> list[Chunk]:
        return list(self._chunks_by_doc.get(document_id, []))
