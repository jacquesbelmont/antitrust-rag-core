from __future__ import annotations

import math
from dataclasses import dataclass

from app.domain.entities import Chunk, RetrievedChunk
from app.domain.ports import VectorStore


@dataclass
class _Record:
    chunk: Chunk
    vector: list[float]


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._records: list[_Record] = []

    async def upsert_chunks(self, *, chunks: list[Chunk]) -> None:
        # For PoC: append-only. In production you'd index by chunk.id and update.
        for c in chunks:
            vec = c.metadata.get("vector")
            if not isinstance(vec, list):
                raise ValueError("chunk missing embedded vector")
            self._records.append(_Record(chunk=c, vector=vec))

    async def search(self, *, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        scored: list[RetrievedChunk] = []
        for r in self._records:
            score = _cosine_similarity(query_vector, r.vector)
            scored.append(RetrievedChunk(chunk=r.chunk, score=score))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom == 0.0:
        return 0.0
    return dot / denom
