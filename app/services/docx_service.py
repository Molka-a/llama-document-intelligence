"""Word (.docx) text and metadata extraction using python-docx."""

import io
import zipfile

import docx
from docx.document import Document as DocxDocument

from app.services.exceptions import InvalidDocumentError
from app.services.text_cleaner import clean_text


def _iter_block_text(document: DocxDocument):
    """Yield paragraph and table text in document order."""
    body = document.element.body
    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            yield docx.text.paragraph.Paragraph(child, document).text
        elif tag == "tbl":
            table = docx.table.Table(child, document)
            for row in table.rows:
                cells = []
                for cell in row.cells:
                    value = cell.text.strip()
                    # Merged cells repeat the same text; keep it once
                    if value and (not cells or cells[-1] != value):
                        cells.append(value)
                if cells:
                    yield " | ".join(cells)


def extract_text_from_docx(file_content: bytes, filename: str = "document.docx") -> dict:
    """Extract cleaned text and core metadata from .docx bytes.

    The response has the same shape as the PDF service so the API and
    the later LLM layer can treat both formats identically. Word files
    have no fixed pages, so the whole text is returned as one "page".

    Raises:
        InvalidDocumentError: the bytes are not a readable .docx file.
    """
    if not zipfile.is_zipfile(io.BytesIO(file_content)):
        raise InvalidDocumentError("File is not a valid .docx (not a ZIP container).")

    try:
        document = docx.Document(io.BytesIO(file_content))
    except Exception as exc:
        raise InvalidDocumentError(f"Word document could not be opened: {exc}") from exc

    text = clean_text("\n".join(_iter_block_text(document)))

    props = document.core_properties
    metadata = {
        "title": props.title,
        "author": props.author,
        "subject": props.subject,
        "keywords": props.keywords,
        "created": props.created.isoformat() if props.created else None,
        "modified": props.modified.isoformat() if props.modified else None,
        "last_modified_by": props.last_modified_by,
    }
    metadata = {k: v for k, v in metadata.items() if v}

    return {
        "filename": filename,
        "file_type": "docx",
        "page_count": 1,
        "text": text,
        "pages": [{"page_number": 1, "text": text, "char_count": len(text)}],
        "metadata": metadata,
        "is_empty": not text.strip(),
    }
