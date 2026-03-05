"""
Reranking strategies — BM25 hybrid (default) and cross-encoder (optional).

Two modes are available:

BM25 hybrid (default, always available)
----------------------------------------
Combines dense cosine similarity scores (from Weaviate ANN search) with sparse
BM25 scores computed in-process.  Compensates for dense models' weakness on
exact-match queries (case numbers, article citations, named entities).

formula: final_score = alpha * dense_score + (1 - alpha) * bm25_score

Cross-encoder (optional — requires ``sentence-transformers``)
--------------------------------------------------------------
Scores (query, chunk) pairs jointly using a transformer model, attending to
both texts simultaneously.  Significantly more accurate than BM25 hybrid but
slower (~50 ms/query on CPU for MiniLM-L-6).

Enable by setting CROSS_ENCODER_MODEL in .env, e.g.:
    CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2  # ~80 MB, English
    CROSS_ENCODER_MODEL=BAAI/bge-reranker-base                 # ~270 MB, multilingual

Install:  pip install sentence-transformers

Deduplication
-------------
Both strategies apply Jaccard near-duplicate removal after scoring to prevent
the LLM context window from being filled with repeated overlapping passages.
"""
from __future__ import annotations

import logging
import math
import re

from legal_rag_shared.domain.entities import RetrievedChunk

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Default BM25 tuning constants
_K1 = 1.5
_B = 0.75
_ALPHA = 0.70

# Jaccard similarity above this threshold → treat as near-duplicate.
# 0.70 (rather than 0.85) catches fragments that share most tokens but differ
# only in a few leading/trailing words from the overlap zone.
_DEDUP_THRESHOLD = 0.70


def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    *,
    top_k: int,
    alpha: float = _ALPHA,
    k1: float = _K1,
    b: float = _B,
) -> list[RetrievedChunk]:
    """
    Re-rank *candidates* using hybrid dense + BM25 scoring, then deduplicate.

    Parameters
    ----------
    query:
        Sanitized search query.
    candidates:
        Dense-retrieved chunks with their similarity scores (range 0–1).
    top_k:
        Number of top results to return after reranking and deduplication.
    alpha:
        Dense weight (``1 - alpha`` goes to BM25).
    k1, b:
        BM25 tuning constants.

    Returns
    -------
    list[RetrievedChunk]
        Up to *top_k* chunks sorted by descending hybrid score, with
        near-duplicates removed.
    """
    if not candidates:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return sorted(candidates, key=lambda r: r.score, reverse=True)[:top_k]

    # Pre-compute tokenized docs once (reused for BM25 and deduplication)
    tokenized_docs = [_tokenize(r.chunk.text) for r in candidates]
    N = len(candidates)

    # Document frequency per token across the candidate corpus
    df_map: dict[str, int] = {}
    for doc_tokens in tokenized_docs:
        for token in set(doc_tokens):
            df_map[token] = df_map.get(token, 0) + 1

    avg_dl = sum(len(dt) for dt in tokenized_docs) / N

    # Compute BM25 scores with Robertson-Spärck Jones IDF
    bm25_scores = [
        _bm25_score(query_tokens, doc_tokens, avg_dl, df_map=df_map, N=N, k1=k1, b=b)
        for doc_tokens in tokenized_docs
    ]

    # Normalise BM25 to [0, 1] using min-max scaling
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
    norm_bm25 = [s / max_bm25 for s in bm25_scores]

    # Combine dense + BM25
    scored = [
        RetrievedChunk(
            chunk=r.chunk,
            score=alpha * r.score + (1 - alpha) * bm25,
        )
        for r, bm25 in zip(candidates, norm_bm25)
    ]

    scored.sort(key=lambda r: r.score, reverse=True)

    # Build chunk_id → token-set lookup before deduplication
    id_to_tokens: dict[str, frozenset[str]] = {
        r.chunk.id: frozenset(tokens)
        for r, tokens in zip(candidates, tokenized_docs)
    }

    return _deduplicate(scored, id_to_tokens, threshold=_DEDUP_THRESHOLD)[:top_k]


# ── BM25 helpers ───────────────────────────────────────────────────────────────

def _bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    avg_dl: float,
    *,
    df_map: dict[str, int],
    N: int,
    k1: float,
    b: float,
) -> float:
    """Compute BM25 score for a single document against the query."""
    if not doc_tokens:
        return 0.0

    dl = len(doc_tokens)
    tf_map: dict[str, int] = {}
    for token in doc_tokens:
        tf_map[token] = tf_map.get(token, 0) + 1

    score = 0.0
    for token in set(query_tokens):
        tf = tf_map.get(token, 0)
        if tf == 0:
            continue
        # Robertson-Spärck Jones IDF (smoothed to avoid log(0))
        df = df_map.get(token, 0)
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * dl / max(avg_dl, 1))
        score += idf * (numerator / denominator)

    return score


def _deduplicate(
    scored: list[RetrievedChunk],
    id_to_tokens: dict[str, frozenset[str]],
    *,
    threshold: float,
) -> list[RetrievedChunk]:
    """Remove near-duplicate chunks (Jaccard token overlap ≥ threshold)."""
    kept: list[RetrievedChunk] = []
    kept_token_sets: list[frozenset[str]] = []

    for r in scored:
        tokens = id_to_tokens.get(r.chunk.id, frozenset())
        is_dup = any(
            (len(tokens & existing) / len(tokens | existing)) >= threshold
            for existing in kept_token_sets
            if tokens and existing and (tokens | existing)
        )
        if not is_dup:
            kept.append(r)
            kept_token_sets.append(tokens)

    return kept


def _tokenize(text: str) -> list[str]:
    """Lowercase Unicode word tokenisation."""
    return _WORD_RE.findall(text.lower())


# ── Cross-encoder reranker (optional) ─────────────────────────────────────────

class CrossEncoderReranker:
    """
    Cross-encoder reranker using sentence-transformers.

    Cross-encoders score (query, document) pairs jointly, attending to both
    texts simultaneously.  This is significantly more accurate than BM25
    hybrid scoring but slower (O(n) forward passes per query, ~50 ms/query
    for MiniLM on CPU).

    Recommended models
    ------------------
    cross-encoder/ms-marco-MiniLM-L-6-v2  — ~80 MB, fast, English
    BAAI/bge-reranker-base                 — ~270 MB, multilingual (FR/PT/EN)
    BAAI/bge-reranker-large               — ~1.3 GB, best quality

    Setup
    -----
    1. pip install sentence-transformers
    2. Set CROSS_ENCODER_MODEL=<model-name> in .env
    3. Model downloads automatically on first use (~HuggingFace cache)

    How to change the model
    -----------------------
    Edit CROSS_ENCODER_MODEL in your .env file and restart the API.
    No code changes required.
    """

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers.cross_encoder import CrossEncoder  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for cross-encoder reranking. "
                "Install with: pip install sentence-transformers"
            ) from exc

        self._model_name = model_name
        self._model = CrossEncoder(model_name)
        logger.info("CrossEncoder loaded: %s", model_name)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """
        Score (query, chunk) pairs with the cross-encoder and return top_k.

        Scores are passed through sigmoid so they land in [0, 1], then
        Jaccard deduplication is applied before truncation.

        Parameters
        ----------
        query:
            Sanitized search query.
        candidates:
            Pool of retrieved chunks to rerank.
        top_k:
            Maximum results to return.
        """
        if not candidates:
            return []

        pairs = [(query, r.chunk.text) for r in candidates]
        raw_scores = self._model.predict(pairs)  # numpy float32 array

        def _sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-float(x)))

        scored = [
            RetrievedChunk(chunk=r.chunk, score=_sigmoid(s))
            for r, s in zip(candidates, raw_scores)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)

        # Reuse BM25 deduplication — token overlap still catches repeated passages
        tokenized_docs = [_tokenize(r.chunk.text) for r in candidates]
        id_to_tokens: dict[str, frozenset[str]] = {
            r.chunk.id: frozenset(tokens)
            for r, tokens in zip(candidates, tokenized_docs)
        }

        return _deduplicate(scored, id_to_tokens, threshold=_DEDUP_THRESHOLD)[:top_k]
