from __future__ import annotations

import re

from app.domain.entities import RetrievedChunk


class BM25Reranker:
    """Hybrid dense + sparse reranker using BM25 term scoring.

    Why hybrid reranking matters for legal RAG:
    - Dense retrieval (embeddings) captures semantic similarity.
    - BM25 excels at exact term matches: case numbers, article citations,
      proper nouns (company names, judges), and French legal formulae
      ("attendu que", "article L. 420-1").
    - Combining both with a linear interpolation consistently outperforms
      either alone on domain-specific retrieval benchmarks.

    Parameters:
    - k1 (1.5): term saturation — higher = raw frequency matters more.
    - b (0.75): length normalisation — 0 = no normalisation, 1 = full.
    - alpha (0.7): weight for dense score; (1-alpha) for BM25 score.
      alpha=0.7 means 70% semantic, 30% keyword — tunable per domain.

    Limitations / next steps:
    - IDF is approximated as 1.0 (no corpus statistics at rerank time).
      For production: precompute IDF from the full 430K-doc corpus.
    - For even higher precision: replace with a cross-encoder
      (e.g., ms-marco-MiniLM-L-12-v2) via sentence-transformers.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, alpha: float = 0.7) -> None:
        self._k1 = k1
        self._b = b
        self._alpha = alpha

    def rerank(
        self,
        *,
        query: str,
        retrieved: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not retrieved:
            return []

        query_terms = _tokenize(query)
        if not query_terms:
            return retrieved[:top_k]

        corpus = [_tokenize(r.chunk.text) for r in retrieved]
        avgdl = sum(len(d) for d in corpus) / len(corpus) if corpus else 1.0

        bm25_raw = [
            _bm25_score(query_terms=query_terms, doc_terms=doc, avgdl=avgdl, k1=self._k1, b=self._b)
            for doc in corpus
        ]

        # Normalise BM25 to [0, 1] so it's comparable to cosine similarity scores.
        max_bm25 = max(bm25_raw) if bm25_raw else 1.0
        bm25_norm = [s / max_bm25 if max_bm25 > 0 else 0.0 for s in bm25_raw]

        ranked: list[tuple[float, RetrievedChunk]] = []
        for r, bm25 in zip(retrieved, bm25_norm):
            hybrid = self._alpha * r.score + (1.0 - self._alpha) * bm25
            ranked.append((hybrid, r))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in ranked[:top_k]]


_WORD = re.compile(r"[\w\-]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD.finditer(text)]


def _bm25_score(
    *,
    query_terms: list[str],
    doc_terms: list[str],
    avgdl: float,
    k1: float,
    b: float,
) -> float:
    if not doc_terms:
        return 0.0
    doc_len = len(doc_terms)
    freq: dict[str, int] = {}
    for t in doc_terms:
        freq[t] = freq.get(t, 0) + 1

    score = 0.0
    for term in set(query_terms):
        tf = freq.get(term, 0)
        if tf == 0:
            continue
        idf = 1.0  # simplified; replace with log((N - df + 0.5) / (df + 0.5)) at scale
        tf_norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / max(avgdl, 1)))
        score += idf * tf_norm

    return score
