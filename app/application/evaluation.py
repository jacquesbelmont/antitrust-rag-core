from __future__ import annotations

import re


class EvaluationService:
    """Evaluation stub with basic, explainable metrics.

    Trade-offs:
    - These metrics are not statistically robust; they are intentionally simple.
    - The goal is to demonstrate an evaluation *interface* and a habit of measurement.
    """

    def context_relevance(self, *, query: str, context: str) -> float:
        q = _tokenize(query)
        c = _tokenize(context)
        if not q:
            return 0.0
        overlap = len(q.intersection(c))
        return overlap / max(1, len(q))

    def faithfulness_proxy(self, *, answer: str, context: str) -> float:
        # Proxy: percentage of answer tokens that appear in the context.
        a = _tokenize(answer)
        c = _tokenize(context)
        if not a:
            return 0.0
        supported = len(a.intersection(c))
        return supported / max(1, len(a))


_WORD = re.compile(r"[\w\-]+", re.UNICODE)


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD.finditer(text)}
