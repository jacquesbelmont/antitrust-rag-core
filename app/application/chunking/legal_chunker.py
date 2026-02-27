from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.config import settings


@dataclass(frozen=True)
class ChunkDraft:
    text: str
    hierarchy_path: list[str]


_HEADING_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Document-level: French court decisions and authority rulings
    ("arret", re.compile(
        r"^(arrêt|arret|jugement|ordonnance|décision\s+n°|decision\s+no?\.?\s*\d)",
        re.IGNORECASE,
    )),
    # Title / Titre
    ("title", re.compile(
        r"^(title|titre|título)\s+[ivxlc0-9]+\b",
        re.IGNORECASE,
    )),
    # Chapter / Chapitre
    ("chapter", re.compile(
        r"^(chapter|chapitre|capítulo)\s+[ivxlc0-9]+\b",
        re.IGNORECASE,
    )),
    # Section (EN / FR / PT) — also handles "§ N" used in French competition decisions
    ("section", re.compile(
        r"^(section|seção|§)\s+[0-9ivxlc]+\b",
        re.IGNORECASE,
    )),
    # Article (all jurisdictions)
    ("article", re.compile(
        r"^(article|art\.)\s+[0-9]+",
        re.IGNORECASE,
    )),
]

# Paragraph-level splits: numbered paragraphs, French "Attendu que" / "Considérant que",
# and explicit paragraph/parágrafo keywords.
_PARAGRAPH_MARKER = re.compile(
    r"^\(?\d+\)?\s+"                       # (N) or N  — numbered paragraphs
    r"|^\b(parágrafo|paragraph)\b\s*"       # explicit keyword
    r"|^(attendu\s+que|considérant\s+que|vu\s+l['']article)\b",  # French legal markers
    re.IGNORECASE,
)


def split_legal_text_hierarchical(
    *,
    text: str,
    max_chunk_chars: int | None = None,
    overlap_chars: int | None = None,
) -> list[ChunkDraft]:
    """Split legal text attempting to preserve hierarchy.

    Trade-offs / Why this approach:
    - We avoid naive fixed-size splitting because it destroys legal structure (articles/sections).
    - We also avoid full NLP semantic segmentation to keep the PoC dependency-light.
    - The compromise: detect common legal headings via regex, track a hierarchy stack, and then
      size-bound chunks using a simple overlap for recall.

    Limitations:
    - Regex headings are jurisdiction/language-dependent.
    - For scanned/OCR'd texts, headings may be noisy.
    """

    max_chars = max_chunk_chars or settings.max_chunk_chars
    overlap = overlap_chars if overlap_chars is not None else settings.chunk_overlap_chars

    lines = [ln.rstrip() for ln in text.splitlines()]

    hierarchy: list[str] = []
    blocks: list[tuple[list[str], str]] = []
    current: list[str] = []

    def flush_current() -> None:
        nonlocal current
        if not current:
            return
        block_text = "\n".join(current).strip()
        if block_text:
            blocks.append((hierarchy.copy(), block_text))
        current = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current and current[-1] != "":
                current.append("")
            continue

        heading_level = None
        for level, pattern in _HEADING_PATTERNS:
            if pattern.match(stripped):
                heading_level = level
                break

        if heading_level is not None:
            flush_current()
            _apply_heading(hierarchy, heading_level, stripped)
            continue

        if _PARAGRAPH_MARKER.match(stripped) and current:
            flush_current()

        current.append(stripped)

    flush_current()

    drafts: list[ChunkDraft] = []
    for path, block in blocks:
        drafts.extend(_size_bound(block_text=block, hierarchy_path=path, max_chars=max_chars, overlap=overlap))

    return drafts


def _apply_heading(hierarchy: list[str], heading_level: str, heading_text: str) -> None:
    # "arret" is document-level (above title); treat it as level 0.
    order = ["arret", "title", "chapter", "section", "article"]
    idx = order.index(heading_level)
    while len(hierarchy) > idx:
        hierarchy.pop()
    while len(hierarchy) < idx:
        hierarchy.append("")
    if len(hierarchy) == idx:
        hierarchy.append(heading_text)
        return
    hierarchy[idx] = heading_text


def _size_bound(*, block_text: str, hierarchy_path: list[str], max_chars: int, overlap: int) -> list[ChunkDraft]:
    if len(block_text) <= max_chars:
        return [ChunkDraft(text=block_text, hierarchy_path=hierarchy_path)]

    drafts: list[ChunkDraft] = []
    start = 0
    while start < len(block_text):
        end = min(len(block_text), start + max_chars)
        slice_text = block_text[start:end].strip()
        if slice_text:
            drafts.append(ChunkDraft(text=slice_text, hierarchy_path=hierarchy_path))
        if end == len(block_text):
            break
        start = max(0, end - overlap)

    return drafts
