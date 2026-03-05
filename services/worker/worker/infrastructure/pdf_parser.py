"""
PDF text extraction via PyMuPDF (fitz).

PyMuPDF ≥ 1.23 ships pre-compiled wheels — no system MuPDF installation needed.
The extractor concatenates page texts with double newlines to preserve paragraph
boundaries that the legal chunker relies on.

Security notes
--------------
* `stream=` mode — the PDF bytes are never written to disk in this process.
* PyMuPDF raises `fitz.FileDataError` on corrupt / encrypted files;  callers
  should catch `PDFParseError` which wraps this.
* Input size is NOT validated here — callers are responsible for enforcing an
  upload size limit (e.g. 50 MB) before passing bytes to `extract_text`.
"""
from __future__ import annotations

import fitz  # PyMuPDF


class PDFParseError(Exception):
    """Raised when PDF bytes cannot be parsed into usable text."""


def extract_text(content: bytes) -> str:
    """
    Extract plain text from *content* (raw PDF bytes).

    Pages are joined with ``\\n\\n`` so that paragraph-level chunk boundaries
    are preserved across page breaks.

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
        If PyMuPDF cannot open or parse the document, or if the resulting text
        is empty (e.g. scanned image PDF with no text layer).
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
            "No text layer found in PDF — document may be a scanned image "
            "and requires OCR before ingestion."
        )

    return "\n\n".join(pages).strip()
