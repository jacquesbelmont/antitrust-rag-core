from __future__ import annotations

from app.domain.entities import RetrievedChunk


class ContextFormatter:
    """Formats retrieved chunks into a single context string.

    Trade-offs / Why this approach:
    - Simple concatenation is predictable and transparent.
    - For legal documents you often want strict traceability, so we include hierarchy paths.
    - More advanced approaches (MMR, section-aware ordering, citations) can be added later.
    """

    def format(self, *, retrieved: list[RetrievedChunk]) -> tuple[str, list[dict]]:
        parts: list[str] = []
        context_items: list[dict] = []

        for r in retrieved:
            path = " > ".join(r.chunk.hierarchy_path)
            header = f"[chunk_id={r.chunk.id} score={r.score:.4f} path={path}]".strip()
            parts.append(header)
            parts.append(r.chunk.text)
            parts.append("")

            context_items.append(
                {
                    "chunk_id": r.chunk.id,
                    "document_id": r.chunk.document_id,
                    "score": r.score,
                    "hierarchy_path": r.chunk.hierarchy_path,
                    "text": r.chunk.text,
                }
            )

        return "\n".join(parts).strip(), context_items
