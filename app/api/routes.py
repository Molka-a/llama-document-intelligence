"""Document and job routes."""

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel

from app.schemas.document import UploadResponse
from app.services.docx_service import extract_text_from_docx
from app.services.exceptions import EncryptedDocumentError, InvalidDocumentError
from app.services.job_service import Job, job_manager
from app.services.llm_service import DEFAULT_MODEL, LLMError, list_local_models
from app.services.pdf_service import extract_text_from_pdf

router = APIRouter(prefix="/documents", tags=["Documents"])
jobs_router = APIRouter(prefix="/jobs", tags=["Jobs"])

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

EXTRACTORS = {
    ".pdf": extract_text_from_pdf,
    ".docx": extract_text_from_docx,
}


class JobAccepted(BaseModel):
    job_id: str
    status: str
    status_url: str
    message: str


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

async def _read_document(file: UploadFile) -> dict:
    """Validate the upload and extract its text. Raises HTTPException on bad input."""
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
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File exceeds the 10 MB limit.")

    try:
        return EXTRACTORS[extension](content, filename=filename)
    except EncryptedDocumentError as exc:
        raise HTTPException(422, str(exc)) from exc
    except InvalidDocumentError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


def _ensure_llm_ready() -> None:
    """Fail fast (503) instead of queuing a job that is bound to fail."""
    try:
        models = list_local_models()
    except LLMError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    if DEFAULT_MODEL not in models:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"Model '{DEFAULT_MODEL}' is not installed. Run: ollama pull {DEFAULT_MODEL}",
        )


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF or Word (.docx) file and return its cleaned text and metadata."""
    document = await _read_document(file)
    return {
        "status": "success",
        "message": f"{document['file_type'].upper()} processed successfully",
        "document": document,
    }


@router.post("/extract", status_code=status.HTTP_202_ACCEPTED, response_model=JobAccepted)
async def extract_document(
    file: UploadFile = File(...),
    document_type: Literal["auto", "cv", "cover_letter"] = Query(
        "auto", description="'auto' detects the type; set it to skip classification"),
):
    """Start a structured extraction with the local LLM.

    Returns immediately with a job id (202 Accepted). Poll GET /jobs/{job_id}
    until the status is 'completed' or 'failed'. On CPU, expect several minutes.
    """
    _ensure_llm_ready()
    document = await _read_document(file)

    if document["is_empty"]:
        raise HTTPException(
            422,
            "No text found in the document (it may be a scanned image, which needs OCR).",
        )

    job = job_manager.submit(document["text"], document["filename"], document_type)
    return JobAccepted(
        job_id=job.job_id,
        status=job.status.value,
        status_url=f"/jobs/{job.job_id}",
        message="Extraction started. Poll status_url for the result.",
    )


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

@jobs_router.get("/{job_id}", response_model=Job)
def get_job(job_id: str):
    """Status of an extraction job, with the result once it is completed."""
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown job id '{job_id}'.")
    return job