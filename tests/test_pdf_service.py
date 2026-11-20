import pymupdf
import pytest

from app.services.exceptions import EncryptedDocumentError, InvalidDocumentError
from app.services.pdf_service import extract_text_from_pdf


def test_extract_text_from_pdf(make_pdf):
    result = extract_text_from_pdf(make_pdf(), filename="sample.pdf")

    assert "Llama Document Intelligence" in result["text"]
    assert result["filename"] == "sample.pdf"
    assert result["file_type"] == "pdf"
    assert result["page_count"] == 1
    assert result["pages"][0]["page_number"] == 1
    assert result["is_empty"] is False


def test_blank_pdf_is_flagged_empty():
    doc = pymupdf.open()
    doc.new_page()
    data = doc.tobytes()
    doc.close()

    assert extract_text_from_pdf(data)["is_empty"] is True


def test_non_pdf_bytes_rejected():
    with pytest.raises(InvalidDocumentError):
        extract_text_from_pdf(b"this is not a pdf")


def test_corrupted_pdf_rejected():
    with pytest.raises(InvalidDocumentError):
        extract_text_from_pdf(b"%PDF-1.7\n" + b"\x00garbage" * 50)


def test_encrypted_pdf_rejected(make_pdf):
    doc = pymupdf.open(stream=make_pdf(), filetype="pdf")
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    doc.close()

    with pytest.raises(EncryptedDocumentError):
        extract_text_from_pdf(data)
