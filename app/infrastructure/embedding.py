from __future__ import annotations

import hashlib
from typing import Final


class DeterministicHashEmbedder:
    """Dependency-free embedding stub.

    Trade-offs:
    - Not a real embedding model.
    - Deterministic across runs, so tests and local dev behave predictably.
    - Swappable for OpenAI/Cohere/Instructor/etc. later via the `Embedder` port.
    """

    _dims: Final[int] = 64

    async def embed_text(self, *, text: str) -> list[float]:
        return _hash_to_vec(text, dims=self._dims)

    async def embed_texts(self, *, texts: list[str]) -> list[list[float]]:
        return [_hash_to_vec(t, dims=self._dims) for t in texts]


def _hash_to_vec(text: str, *, dims: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vec: list[float] = []
    for i in range(dims):
        b = digest[i % len(digest)]
        vec.append((b / 255.0) * 2.0 - 1.0)
    return vec
