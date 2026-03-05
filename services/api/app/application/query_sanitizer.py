"""
Query sanitizer — prevents prompt injection and other input abuse.

Prompt injection is the primary LLM-specific security risk for this service.
An attacker who can control the query string passed to the LLM could attempt to
override the system prompt, exfiltrate context chunks, or change the response
format.

Defence strategy (defence-in-depth)
-------------------------------------
1. **Length limit** — truncating before any LLM call; keeps token budgets safe.
2. **Null-byte / control-character stripping** — prevents log-injection and
   encoding tricks that smuggle instructions past string matching.
3. **Injection pattern detection** — a curated list of common meta-instruction
   phrases.  Raises ``PromptInjectionError`` (→ HTTP 422) so the caller knows
   the request was rejected for security reasons, not a generic bad input.
4. **Whitespace normalisation** — collapses redundant whitespace so the chunk
   retrieval and BM25 stages work on clean tokens.

This is not a foolproof defence (prompt injection is an open research problem),
but it significantly raises the bar for casual attackers.
"""
from __future__ import annotations

import re
import unicodedata

from app.application.errors import PromptInjectionError, ValidationError

# Hard upper limit on raw query length (chars).
_MAX_QUERY_CHARS = 1_000

# Patterns characteristic of prompt injection / jailbreak attempts.
# All comparisons are done on a lower-cased, whitespace-normalised copy.
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all|your\s+instructions?)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?!a\s+legal)", re.IGNORECASE),   # "you are now DAN"
    re.compile(r"\bsystem\s*:\s*\[", re.IGNORECASE),                  # system: [ … ]
    re.compile(r"\b(assistant|human|user)\s*:\s*\n", re.IGNORECASE),  # role-flip markers
    re.compile(r"<\s*/?(?:system|user|assistant)\s*>", re.IGNORECASE),# XML role tags
    re.compile(r"print\s+your\s+(system\s+)?prompt", re.IGNORECASE),
    re.compile(r"reveal\s+(your\s+)?(system\s+)?prompt", re.IGNORECASE),
    re.compile(r"output\s+your\s+(full\s+)?instructions?", re.IGNORECASE),
    re.compile(r"act\s+as\s+(?!a\s+legal)", re.IGNORECASE),           # "act as [persona]"
    re.compile(r"do\s+anything\s+now", re.IGNORECASE),                 # DAN variant
    re.compile(r"\bDAN\b"),                                             # classic DAN
]

# Characters that must not appear in a query (null bytes, ANSI escape seqs, etc.)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize(query: str) -> str:
    """
    Validate and sanitize *query* for use in an LLM prompt.

    Parameters
    ----------
    query:
        Raw query string from the HTTP request body.

    Returns
    -------
    str
        Sanitized query, safe for embedding and LLM use.

    Raises
    ------
    ValidationError
        If the query is empty after sanitization or exceeds the length limit.
    PromptInjectionError
        If the query matches a known injection pattern.
    """
    if not isinstance(query, str):
        raise ValidationError("Query must be a string.")

    # 1. Length check (before any expensive processing)
    if len(query) > _MAX_QUERY_CHARS:
        raise ValidationError(
            f"Query too long: {len(query)} chars (max {_MAX_QUERY_CHARS})."
        )

    # 2. Strip control characters
    query = _CONTROL_CHAR_RE.sub("", query)

    # 3. Unicode normalisation (NFC) — prevents homoglyph bypasses
    query = unicodedata.normalize("NFC", query)

    # 4. Collapse redundant whitespace
    query = " ".join(query.split())

    # 5. Empty check (after stripping)
    if not query:
        raise ValidationError("Query must not be empty.")

    # 6. Injection pattern detection
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(query):
            raise PromptInjectionError(
                "The query contains patterns that are not permitted. "
                "Please submit a plain legal question."
            )

    return query
