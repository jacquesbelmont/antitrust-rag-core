"""
Lightweight evaluation metrics — surfaced in the search API response.

These are deliberately simple heuristics (token overlap) rather than model-based
metrics (BERTScore, G-Eval).  They provide quick, interpretable feedback during
development and demos without adding model dependencies.

Metrics
-------
context_relevance
    How much of the query is "covered" by the retrieved context.
    = |query_tokens ∩ context_tokens| / max(|query_tokens|, 1)
    Range: [0, 1], higher is better.

faithfulness_proxy
    How much of the generated answer is grounded in the context.
    = |answer_tokens ∩ context_tokens| / max(|answer_tokens|, 1)
    Range: [0, 1], higher is better.
"""
from __future__ import annotations

import re

_WORD_RE = re.compile(r"\w+", re.UNICODE)


class EvaluationService:
    def evaluate(
        self,
        query: str,
        context: str,
        answer: str,
        *,
        chunks_retrieved: int,
        reranking_enabled: bool,
    ) -> dict:
        """
        Return an evaluation payload suitable for inclusion in the HTTP response.
        """
        q_tokens = set(_WORD_RE.findall(query.lower()))
        c_tokens = set(_WORD_RE.findall(context.lower()))
        a_tokens = set(_WORD_RE.findall(answer.lower()))

        context_relevance = (
            len(q_tokens & c_tokens) / max(len(q_tokens), 1)
        )
        faithfulness_proxy = (
            len(a_tokens & c_tokens) / max(len(a_tokens), 1)
        )

        return {
            "context_relevance": round(context_relevance, 4),
            "faithfulness_proxy": round(faithfulness_proxy, 4),
            "chunks_retrieved": chunks_retrieved,
            "reranking_enabled": reranking_enabled,
        }
