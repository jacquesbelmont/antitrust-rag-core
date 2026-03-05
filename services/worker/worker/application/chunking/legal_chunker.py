"""
Hierarchical French legal text splitter.

Detects structural headings and paragraph-level breaks to produce semantically
coherent chunks that preserve the document's hierarchy path.

Supported document types
------------------------
French:
* Code / statute (TITRE, CHAPITRE, SECTION, Article)
* Court decisions / arrêts (ARRÊT, JUGEMENT, SUR LA…, ATTENDU QUE, PAR CES MOTIFS)
* Administrative decisions (DÉCISION, ORDONNANCE, CONSIDÉRANT QUE, I./II./A./B.)

Portuguese / Brazilian:
* Lei / Decreto / Resolução / Portaria / Acórdão
* Artigo N / Art. N° (Brazilian Civil and Administrative law)
* Seção, Capítulo, Título (Brazilian codes)
* CONSIDERANDO (Brazilian "Whereas" preamble)
* I - , II - (Roman numeral list items common in Brazilian law)
* RESOLVE, DETERMINA (Brazilian administrative operative parts)

Generic:
* Numbered sections (§ N, I., II., A., B.)

Design trade-offs
-----------------
* Regex-based heading detection — avoids heavy NLP dependencies; predictable on
  formal legal language where headings follow strict typographic conventions.
* Overlap snapped to word boundary — the previous implementation could start
  chunks mid-word (e.g. "ans des conditions…" instead of "dans des conditions").
* Minimum chunk length — fragments shorter than ``_MIN_CHUNK_CHARS`` are
  discarded before embedding to prevent low-quality vectors polluting search.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ── Heading patterns (order = hierarchy level, most general first) ─────────────
#
# The *index* of each pattern becomes the heading's nesting depth:
#   0 = document-level (ARRÊT / DÉCISION / …)
#   1 = TITRE
#   2 = CHAPITRE
#   3 = SECTION
#   4 = Article / ART.
#   5 = Roman-numeral part (I., II., III., …)
#   6 = SUR LA … / PAR CES MOTIFS (court-decision section markers)
#   7 = ATTENDU QUE / CONSIDÉRANT QUE / VU (recitals)
#   8 = Alphabetic subsection (A., B., C., …)
#   9 = § N (paragraph numbering)
_HEADING_PATTERNS: list[re.Pattern[str]] = [
    # 0 — Document-level keyword (French + Portuguese/Brazilian)
    re.compile(
        r"^\s*(ARRÊT|ARRET|JUGEMENT|ORDONNANCE|DÉCISION|DECISION"
        r"|ACÓRDÃO|ACORDAO|RESOLUÇÃO|RESOLUCAO|PORTARIA|DECRETO|LEI\s+N"
        r"|INSTRUÇÃO\s+NORMATIVA|INSTRUCAO\s+NORMATIVA)\b",
        re.IGNORECASE | re.MULTILINE,
    ),
    # 1 — TITRE / TÍTULO [roman/arabic]
    re.compile(
        r"^\s*(TITRE|TÍTULO|TITULO)\s+[IVXLCDM\d]+",
        re.IGNORECASE | re.MULTILINE,
    ),
    # 2 — CHAPITRE / CAPÍTULO [roman/arabic]
    re.compile(
        r"^\s*(CHAPITRE|CAPÍTULO|CAPITULO)\s+[IVXLCDM\d]+",
        re.IGNORECASE | re.MULTILINE,
    ),
    # 3 — SECTION / SEÇÃO [roman/arabic]
    re.compile(
        r"^\s*(SECTION|SEÇÃO|SECAO|SEÇÃO)\s+[IVXLCDM\d]+",
        re.IGNORECASE | re.MULTILINE,
    ),
    # 4 — Article / Artigo / Art. N
    re.compile(
        r"^\s*(Article|Artigo|ART\.?)\s*\.?\s*\d+",
        re.IGNORECASE | re.MULTILINE,
    ),
    # 5 — Roman-numeral section: "I.", "II.", "IV.", … (up to XIX)
    re.compile(
        r"^\s*(?:I{1,3}|IV|VI{0,3}|IX|XI{1,3}|XIV|XV|XVI{0,3}|XIX)\.\s+\S",
        re.MULTILINE,
    ),
    # 6 — Brazilian roman-numeral list items: "I -", "II -", "III –"
    re.compile(
        r"^\s*(?:I{1,3}|IV|VI{0,3}|IX|XI{1,3}|XIV|XV)\s*[-–]\s+\S",
        re.MULTILINE,
    ),
    # 7 — French court/admin section markers
    re.compile(
        r"^\s*(SUR\s+(?:LA|LE|LES|L')|PAR CES MOTIFS)\b",
        re.IGNORECASE | re.MULTILINE,
    ),
    # 8 — Brazilian administrative operative parts
    re.compile(
        r"^\s*(RESOLVE|DETERMINA|CONSIDERANDO|DELIBERA|APROVA)\b",
        re.IGNORECASE | re.MULTILINE,
    ),
    # 9 — Recitals (French + Portuguese)
    re.compile(
        r"^\s*(Attendu que|Considérant que|Vu que|CONSIDERANT|VU\s*:|Tendo em vista|Considerando que)\b",
        re.IGNORECASE | re.MULTILINE,
    ),
    # 10 — Single uppercase letter subsection: "A.", "B.", … followed by capital
    re.compile(r"^\s*[A-Z]\.\s+[A-ZÀÂÉÈÊËÎÏÔÙÛÜŸÆŒÁÉÍÓÚÃÕÂÊÎÔÛÇÀÈÌÒÙA-Z\d\(]", re.MULTILINE),
    # 11 — Paragraph § N
    re.compile(r"^\s*§\s*\d+", re.MULTILINE),
]

# ── Paragraph-level break patterns ────────────────────────────────────────────
_PARA_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*\(\d+\)\s", re.MULTILINE),           # (1) (2) …
    re.compile(
        r"^\s*(Attendu que|Considérant que|Vu que|Considerando que|Tendo em vista)\b",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(r"\n{2,}"),                                  # blank lines
]

# Heading level map — informational only (not used by logic)
_HEADING_LEVELS = {
    0: "DOCUMENT", 1: "TÍTULO", 2: "CAPÍTULO", 3: "SEÇÃO",
    4: "ARTIGO", 5: "PARTE_ROMANA", 6: "ITEM_ROMANO", 7: "MOTIF",
    8: "OPERATIVO", 9: "RECITAL", 10: "SUBSEÇÃO", 11: "PARÁGRAFO",
}

# Chunks shorter than this are skipped (usually artefacts of overlap/list items)
_MIN_CHUNK_CHARS = 80


@dataclass(frozen=True)
class ChunkDraft:
    """Intermediate chunk before domain IDs are assigned."""
    text: str
    hierarchy_path: list[str]


def split_legal_text_hierarchical(
    text: str,
    max_chunk_chars: int = 1400,
    chunk_overlap_chars: int = 120,
) -> list[ChunkDraft]:
    """
    Split *text* into hierarchical chunks.

    Parameters
    ----------
    text:
        Full document text (UTF-8 string).
    max_chunk_chars:
        Hard upper limit on chunk character length.
    chunk_overlap_chars:
        Number of trailing characters carried into the next split when a block
        is too large to fit in a single chunk.  The carry-over is snapped to a
        word boundary so chunks never start mid-word.

    Returns
    -------
    list[ChunkDraft]
        Ordered list of chunks; each carries the hierarchy_path active at that
        point in the document.  Chunks shorter than ``_MIN_CHUNK_CHARS`` are
        excluded.
    """
    if not text:
        return []

    segments = _segment_by_headings(text)
    chunks: list[ChunkDraft] = []

    for segment_text, path in segments:
        segment_text = segment_text.strip()
        if len(segment_text) < _MIN_CHUNK_CHARS:
            continue

        if len(segment_text) <= max_chunk_chars:
            chunks.append(ChunkDraft(text=segment_text, hierarchy_path=list(path)))
        else:
            for sub in _size_bound(segment_text, path, max_chunk_chars, chunk_overlap_chars):
                chunks.append(sub)

    return chunks


# ── Internal helpers ───────────────────────────────────────────────────────────

def _segment_by_headings(text: str) -> list[tuple[str, list[str]]]:
    """
    Find all heading positions, split the text at those boundaries, and return
    (segment_text, hierarchy_path) pairs.
    """
    events: list[tuple[int, str, int]] = []
    for level, pattern in enumerate(_HEADING_PATTERNS):
        for m in pattern.finditer(text):
            newline = text.find("\n", m.start())
            heading_line = text[m.start(): newline if newline != -1 else m.start() + 120]
            events.append((m.start(), heading_line.strip(), level))

    if not events:
        return [(text, [])]

    events.sort(key=lambda e: e[0])

    hierarchy_stack: list[str] = []
    segments: list[tuple[str, list[str]]] = []

    first_pos = events[0][0]
    if first_pos > 0:
        preamble = text[:first_pos].strip()
        if preamble:
            segments.append((preamble, []))

    for i, (pos, heading, level) in enumerate(events):
        hierarchy_stack = hierarchy_stack[:level]
        hierarchy_stack.append(heading)

        end_pos = events[i + 1][0] if i + 1 < len(events) else len(text)
        segment_text = text[pos:end_pos]
        segments.append((segment_text, list(hierarchy_stack)))

    return segments


def _size_bound(
    text: str,
    path: list[str],
    max_chars: int,
    overlap: int,
) -> list[ChunkDraft]:
    """
    Split an oversized segment into max_chars pieces with *overlap* carry-over.

    The carry-over position is snapped forward to the next word boundary so
    chunks never start mid-word (fixes the "ans des conditions…" fragment bug).
    Tries to break at paragraph boundaries first; falls back to hard cuts.
    """
    chunks: list[ChunkDraft] = []

    break_positions: list[int] = [0]
    for pattern in _PARA_PATTERNS:
        for m in pattern.finditer(text):
            break_positions.append(m.start())
    break_positions.append(len(text))
    break_positions = sorted(set(break_positions))

    current_start = 0
    while current_start < len(text):
        target_end = current_start + max_chars

        if target_end >= len(text):
            piece = text[current_start:].strip()
            if len(piece) >= _MIN_CHUNK_CHARS:
                chunks.append(ChunkDraft(text=piece, hierarchy_path=list(path)))
            break

        # Find the largest paragraph break ≤ target_end
        best_break = target_end
        for bp in break_positions:
            if current_start < bp <= target_end:
                best_break = bp

        piece = text[current_start:best_break].strip()
        if len(piece) >= _MIN_CHUNK_CHARS:
            chunks.append(ChunkDraft(text=piece, hierarchy_path=list(path)))

        # Advance with overlap — snap to the next word boundary so the
        # following chunk never starts mid-word.
        new_start = max(current_start + 1, best_break - overlap)
        while new_start < len(text) and not text[new_start].isspace():
            new_start += 1
        while new_start < len(text) and text[new_start].isspace():
            new_start += 1
        current_start = new_start

    return chunks
