"""Document upload routes."""

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.schemas.document import UploadResponse
from app.services.docx_service import extract_text_from_docx
from app.services.exceptions import EncryptedDocumentError, InvalidDocumentError
from app.services.pdf_service import extract_text_from_pdf

router = APIRouter(prefix="/documents", tags=["Documents"])

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

EXTRACTORS = {
    ".pdf": extract_text_from_pdf,
    ".docx": extract_text_from_docx,
}


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF or Word (.docx) file and return its cleaned text and metadata."""
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    if extension not in EXTRACTORS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{extension or 'none'}'. "
                   f"Allowed: {', '.join(EXTRACTORS)}",
        )

    content = await file.read()

    if not content:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The uploaded file is empty.")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            "File exceeds the 10 MB limit.",
        )

    try:
        document = EXTRACTORS[extension](content, filename=filename)
    except EncryptedDocumentError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except InvalidDocumentError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    return {
        "status": "success",
        "message": f"{extension[1:].upper()} processed successfully",
        "document": document,
    }