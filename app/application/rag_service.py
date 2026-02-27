from __future__ import annotations

from app.application.context_formatting import ContextFormatter
from app.application.evaluation import EvaluationService
from app.application.reranking import BM25Reranker
from app.application.retrieval_service import RetrievalService
from app.domain.ports import LLMClient


class RAGService:
    """Orchestrates the full RAG pipeline:

        embed(query) → dense retrieval → BM25 rerank → context formatting → LLM → evaluation

    The reranker is optional: when None, dense retrieval results go directly to the LLM.
    This lets us A/B test reranking impact on answer quality without code changes.
    """

    def __init__(
        self,
        *,
        retrieval: RetrievalService,
        formatter: ContextFormatter,
        llm: LLMClient,
        evaluation: EvaluationService,
        reranker: BM25Reranker | None = None,
    ) -> None:
        self._retrieval = retrieval
        self._formatter = formatter
        self._llm = llm
        self._evaluation = evaluation
        self._reranker = reranker

    async def answer(self, *, query: str, top_k: int) -> tuple[str, list[dict], dict]:
        # Over-retrieve when reranking: more candidates → better final precision.
        candidate_k = top_k * 3 if self._reranker else top_k
        retrieved = await self._retrieval.retrieve(query=query, top_k=candidate_k)

        if self._reranker:
            retrieved = self._reranker.rerank(query=query, retrieved=retrieved, top_k=top_k)

        context_text, context_items = self._formatter.format(retrieved=retrieved)
        prompt = _build_prompt(query=query, context=context_text)
        answer = await self._llm.generate(prompt=prompt)

        eval_payload = {
            "context_relevance": self._evaluation.context_relevance(query=query, context=context_text),
            "faithfulness_proxy": self._evaluation.faithfulness_proxy(answer=answer, context=context_text),
            "chunks_retrieved": len(retrieved),
            "reranking_enabled": self._reranker is not None,
        }

        return answer, context_items, eval_payload


def _build_prompt(*, query: str, context: str) -> str:
    return (
        "You are a careful legal assistant specialising in antitrust and competition law. "
        "Answer ONLY using the provided context. "
        "Cite the relevant chunk paths when you quote a passage. "
        "If the context is insufficient, say explicitly that you don't know.\n\n"
        f"QUESTION:\n{query}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "ANSWER:\n"
    )
