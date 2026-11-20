import pytest

from app.services.docx_service import extract_text_from_docx
from app.services.exceptions import InvalidDocumentError


def test_extract_paragraphs_and_metadata(make_docx):
    result = extract_text_from_docx(make_docx(), filename="cv.docx")

    assert result["text"] == "Molka Amara\nData Scientist"
    assert result["file_type"] == "docx"
    assert result["filename"] == "cv.docx"
    assert result["metadata"]["author"] == "Tester"
    assert result["is_empty"] is False


def test_extract_table_in_order(make_docx):
    data = make_docx(
        paragraphs=["Skills"],
        table=[["Python", "Advanced"], ["FastAPI", "Intermediate"]],
    )
    text = extract_text_from_docx(data)["text"]

    assert text == "Skills\nPython | Advanced\nFastAPI | Intermediate"


def test_empty_docx_flagged(make_docx):
    assert extract_text_from_docx(make_docx(paragraphs=[]))["is_empty"] is True


def test_invalid_docx_rejected():
    with pytest.raises(InvalidDocumentError):
        extract_text_from_docx(b"not a word file")
