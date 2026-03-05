"""
PDF text extraction for the API service (synchronous, called at upload time).

PyMuPDF ≥ 1.23 ships pre-compiled wheels — no system MuPDF installation needed.
The extractor operates in stream mode: bytes are never written to disk.

Why extract in the API instead of the worker?
----------------------------------------------
Extracting text synchronously in the API (before the 202 response) lets us
return a clear 422 immediately for corrupt/encrypted/image PDFs, rather than
discovering the error asynchronously in the worker after the user has already
received a 202 and has no way to know it silently failed.

The worker receives pre-extracted text via PostgreSQL and only performs the
expensive chunking + embedding phase — the same code path as the JSON upload.

Security notes
--------------
* ``stream=`` mode — the PDF bytes are never written to disk in this process.
* PyMuPDF raises ``fitz.FileDataError`` on corrupt / encrypted files; callers
  should catch ``PDFParseError`` which wraps it.
* Input size is NOT validated here — the route enforces a 50 MB cap before
  calling ``extract_text``.
"""
from __future__ import annotations

import fitz  # PyMuPDF


class PDFParseError(Exception):
    """Raised when PDF bytes cannot be parsed into usable text."""


def extract_text(content: bytes) -> str:
    """
    Extract plain text from *content* (raw PDF bytes).

    Pages are joined with ``\\n\\n`` to preserve paragraph-level boundaries
    that the legal chunker relies on for hierarchy detection.

    Parameters
    ----------
    content:
        Raw bytes of the PDF file.

    Returns
    -------
    str
        Concatenated page texts, stripped of leading/trailing whitespace.

    Raises
    ------
    PDFParseError
        If PyMuPDF cannot open or parse the document, or if the extracted
        text is empty (e.g. scanned image PDF with no text layer).
    """
    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise PDFParseError(f"Cannot open PDF: {exc}") from exc

    try:
        pages: list[str] = []
        for page in doc:
            page_text = page.get_text()
            if page_text.strip():
                pages.append(page_text)
    finally:
        doc.close()

    if not pages:
        raise PDFParseError(
            "No text layer found in PDF — the document may be a scanned image "
            "and requires OCR preprocessing before ingestion."
        )

    return "\n\n".join(pages).strip()
