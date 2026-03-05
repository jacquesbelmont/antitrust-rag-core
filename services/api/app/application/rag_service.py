"""
RAG pipeline orchestrator.

Pipeline
--------
1. **Sanitize**  — Strip/reject injection attempts (via query_sanitizer).
2. **Retrieve**  — Embed query → ANN search in Weaviate.
3. **Rerank**    — Optional hybrid BM25 reranking + near-duplicate removal.
4. **Filter**    — Drop chunks below the minimum relevance threshold.
5. **Format**    — Build LLM prompt from top-k chunks (with character budget).
6. **Generate**  — Call Ollama with system prompt + numbered context.
7. **Evaluate**  — Compute lightweight relevance metrics.

Anti-hallucination measures
----------------------------
* System prompt instructs the model to cite chunk numbers ([1], [2], …) for
  every claim and to explicitly say when the answer is not in the context.
* Score threshold (``_MIN_SCORE``) removes low-relevance chunks before they
  reach the LLM — reducing the risk of the model "confabulating" from
  tangentially related text.
* ``temperature=0.0`` makes generation deterministic.

Security
--------
* The query is sanitized before embedding (prevents injection through the
  vector store query path) and before prompt construction.
* System and user content are clearly delimited in the prompt.
"""
from __future__ import annotations

import logging

from legal_rag_shared.domain.ports import Embedder, LLMClient, VectorStore
from app.application.context_formatting import ContextFormatter
from app.application.errors import StorageError
from app.application.evaluation import EvaluationService
from app.application.query_sanitizer import sanitize
from app.application.reranking import CrossEncoderReranker, rerank
from app.application.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

# Minimum hybrid score (dense 70% + BM25 30%) for a chunk to enter the LLM
# context.  Chunks below this are likely off-topic and increase hallucination.
_MIN_SCORE: float = 0.25

# System prompt — strict grounding instructions (language-agnostic).
_SYSTEM_PROMPT = (
    "You are an expert legal document assistant.\n"
    "Your task is to answer the question using ONLY the numbered excerpts provided below.\n\n"
    "Strict rules:\n"
    "1. Cite each excerpt you use by its number in brackets: [1], [2], etc.\n"
    "2. Do NOT invent any information not present in the excerpts.\n"
    "3. If the excerpts do not contain the answer, respond exactly: "
    "\"The provided excerpts do not contain enough information to answer this question.\"\n"
    "4. ALWAYS respond in the same language as the question asked.\n"
    "5. Be concise and structured. No greetings or preamble."
)

_NO_CONTEXT_ANSWER = (
    "No sufficiently relevant excerpts were found in the document corpus "
    "to answer this question."
)


class RAGService:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        formatter: ContextFormatter,
        llm: LLMClient,
        evaluator: EvaluationService,
        *,
        enable_reranking: bool = True,
        cross_encoder: CrossEncoderReranker | None = None,
    ) -> None:
        self._retrieval = retrieval_service
        self._formatter = formatter
        self._llm = llm
        self._evaluator = evaluator
        self._enable_reranking = enable_reranking
        self._cross_encoder = cross_encoder

    async def answer(
        self,
        raw_query: str,
        top_k: int,
    ) -> tuple[str, list[dict], dict]:
        """
        Run the full RAG pipeline.

        Parameters
        ----------
        raw_query:
            Unsanitized query string from the HTTP request.
        top_k:
            Number of chunks to include in the final context.

        Returns
        -------
        tuple[str, list[dict], dict]
            ``(answer_text, context_items, evaluation_payload)``
        """
        # 1. Sanitize
        query = sanitize(raw_query)

        # 2. Retrieve (over-fetch when reranking for wider candidate pool)
        candidate_k = top_k * 3 if self._enable_reranking else top_k
        candidates = await self._retrieval.retrieve(query, top_k=candidate_k)

        # 3. Rerank (includes near-duplicate removal)
        # Cross-encoder (if configured) > BM25 hybrid > no reranking
        if self._enable_reranking and candidates:
            if self._cross_encoder is not None:
                results = self._cross_encoder.rerank(query, candidates, top_k=top_k)
            else:
                results = rerank(query, candidates, top_k=top_k)
        else:
            results = candidates[:top_k]

        # 4. Filter — drop chunks below minimum relevance threshold
        results = [r for r in results if r.score >= _MIN_SCORE]

        if not results:
            logger.info("No chunks above score threshold — returning no-context answer")
            evaluation = self._evaluator.evaluate(
                query,
                "",
                _NO_CONTEXT_ANSWER,
                chunks_retrieved=0,
                reranking_enabled=self._enable_reranking,
            )
            return _NO_CONTEXT_ANSWER, [], evaluation

        # 5. Format context (numbered citations + character budget)
        context_str, context_items = self._formatter.format(results)

        # 6. Generate — system prompt sent as separate role via Chat API so the
        # model honours grounding constraints at the attention level.
        prompt = (
            f"Question : {query}\n\n"
            f"Extraits disponibles :\n{context_str}\n\n"
            "Réponse :"
        )
        try:
            answer_text = await self._llm.generate(prompt, system=_SYSTEM_PROMPT)
        except Exception as exc:
            logger.exception("LLM generation failed", extra={"error_type": type(exc).__name__})
            raise StorageError(f"LLM generation failed: {exc}") from exc

        # 7. Evaluate
        evaluation = self._evaluator.evaluate(
            query,
            context_str,
            answer_text,
            chunks_retrieved=len(results),
            reranking_enabled=self._enable_reranking,
        )

        return answer_text, context_items, evaluation
