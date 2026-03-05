"""
Context formatter — converts RetrievedChunk objects into a prompt-ready string
and a structured list for the HTTP response.

Formatting strategy
-------------------
Each chunk is numbered [1], [2], … so the LLM can cite sources precisely
(e.g. "Selon [2], …").  A character budget prevents sending huge prompts to
Ollama that would be silently truncated at the model's context window.
"""
from __future__ import annotations

from legal_rag_shared.domain.entities import RetrievedChunk

# Approximate character budget for the context block sent to the LLM.
# llama3.1 has a 128k token context; at ~4 chars/token this is ~12k tokens —
# well within limits while still providing rich context.
_DEFAULT_MAX_CHARS = 12_000


class ContextFormatter:
    """Serialises retrieved chunks for LLM prompt construction and API response."""

    def __init__(self, max_chars: int = _DEFAULT_MAX_CHARS) -> None:
        self._max_chars = max_chars

    def format(
        self,
        retrieved: list[RetrievedChunk],
    ) -> tuple[str, list[dict]]:
        """
        Parameters
        ----------
        retrieved:
            Chunks to include in the context, in descending score order.

        Returns
        -------
        tuple[str, list[dict]]
            ``(prompt_context_string, structured_context_items)``

            *prompt_context_string* — numbered list of chunks ready for an LLM
            prompt.  Each chunk is labelled [N] with its hierarchy path so the
            model can cite sources precisely.

            *structured_context_items* — list of dicts for the API JSON response,
            one per chunk (always contains the full text, regardless of budget).
        """
        lines: list[str] = []
        items: list[dict] = []
        total_chars = 0

        for i, r in enumerate(retrieved, 1):
            path_str = " > ".join(r.chunk.hierarchy_path) if r.chunk.hierarchy_path else "(racine)"
            chunk_text = r.chunk.text

            # Budget guard: truncate chunk text if needed, skip if nothing fits
            remaining_budget = self._max_chars - total_chars
            if remaining_budget < 80:
                items.append(_make_item(r))
                continue

            if len(chunk_text) > remaining_budget:
                chunk_text = chunk_text[:remaining_budget] + "…"

            header = f"[{i}] {path_str}"
            lines.append(header)
            lines.append(chunk_text)
            lines.append("\n---")  # clear visual boundary between chunks

            total_chars += len(chunk_text)
            items.append(_make_item(r))

        return "\n".join(lines).strip(), items


def _make_item(r: RetrievedChunk) -> dict:
    return {
        "chunk_id": r.chunk.id,
        "document_id": r.chunk.document_id,
        "score": round(r.score, 4),
        "hierarchy_path": r.chunk.hierarchy_path,
        "text": r.chunk.text,  # always full text in API response
    }
