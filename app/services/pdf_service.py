"""PDF text and metadata extraction using PyMuPDF."""

import pymupdf

from app.services.exceptions import EncryptedDocumentError, InvalidDocumentError
from app.services.text_cleaner import clean_text


def extract_text_from_pdf(file_content: bytes, filename: str = "document.pdf") -> dict:
    """Extract cleaned text, per-page content and metadata from PDF bytes.

    Raises:
        InvalidDocumentError: the bytes are not a readable PDF.
        EncryptedDocumentError: the PDF is password-protected.
    """
    if not file_content.startswith(b"%PDF"):
        raise InvalidDocumentError("File is not a valid PDF (missing %PDF header).")

    try:
        doc = pymupdf.open(stream=file_content, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several error types
        raise InvalidDocumentError(f"PDF could not be opened: {exc}") from exc

    with doc:
        if doc.needs_pass:
            raise EncryptedDocumentError("PDF is password-protected.")

        pages = []
        for index, page in enumerate(doc, start=1):
            page_text = clean_text(page.get_text("text"))
            pages.append({
                "page_number": index,
                "text": page_text,
                "char_count": len(page_text),
            })

        full_text = "\n\n".join(p["text"] for p in pages if p["text"])
        metadata = {k: v for k, v in (doc.metadata or {}).items() if v}

        return {
            "filename": filename,
            "file_type": "pdf",
            "page_count": doc.page_count,
            "text": full_text,
            "pages": pages,
            "metadata": metadata,
            # No text at all usually means a scanned PDF that needs OCR
            "is_empty": not full_text.strip(),
        }
