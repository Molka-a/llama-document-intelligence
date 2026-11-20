from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_pdf(make_pdf):
    r = client.post("/documents/upload", files={"file": ("a.pdf", make_pdf(), "application/pdf")})
    assert r.status_code == 200
    assert r.json()["document"]["file_type"] == "pdf"


def test_upload_docx(make_docx):
    r = client.post("/documents/upload", files={"file": ("cv.docx", make_docx(), DOCX_MIME)})
    assert r.status_code == 200
    assert "Molka Amara" in r.json()["document"]["text"]


def test_unsupported_extension():
    r = client.post("/documents/upload", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_empty_file():
    r = client.post("/documents/upload", files={"file": ("a.pdf", b"", "application/pdf")})
    assert r.status_code == 400


def test_fake_pdf_rejected():
    r = client.post("/documents/upload", files={"file": ("a.pdf", b"not a pdf", "application/pdf")})
    assert r.status_code == 400
    assert "not a valid PDF" in r.json()["detail"]
