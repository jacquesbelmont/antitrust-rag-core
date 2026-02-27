from app.application.reranking import BM25Reranker
from app.domain.entities import Chunk, RetrievedChunk


def _make_retrieved(chunk_id: str, text: str, score: float) -> RetrievedChunk:
    chunk = Chunk(id=chunk_id, document_id="d1", text=text, index=0, hierarchy_path=[], metadata={})
    return RetrievedChunk(chunk=chunk, score=score)


def test_reranker_promotes_exact_term_match():
    reranker = BM25Reranker(alpha=0.5)  # equal weight to dense and BM25

    # Two chunks with same dense score; only chunk_b contains the query term.
    chunk_a = _make_retrieved("a", "general antitrust principles apply", score=0.9)
    chunk_b = _make_retrieved("b", "article L. 420-1 prohibits cartels", score=0.7)

    results = reranker.rerank(
        query="article L. 420-1",
        retrieved=[chunk_a, chunk_b],
        top_k=2,
    )

    # chunk_b should rank first due to BM25 boosting the exact article reference
    assert results[0].chunk.id == "b"


def test_reranker_respects_top_k():
    reranker = BM25Reranker()
    chunks = [_make_retrieved(str(i), f"chunk text {i}", score=float(i) / 10) for i in range(10)]
    results = reranker.rerank(query="chunk", retrieved=chunks, top_k=3)
    assert len(results) == 3


def test_reranker_empty_input():
    reranker = BM25Reranker()
    assert reranker.rerank(query="anything", retrieved=[], top_k=5) == []
