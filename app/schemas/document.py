"""API-level schemas: extraction output and upload response."""

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from app.schemas.cover_letter import CoverLetterDocument
from app.schemas.cv import CVDocument


class DocumentType(str, Enum):
    CV = "cv"
    COVER_LETTER = "cover_letter"
    OTHER = "other"


class Page(BaseModel):
    page_number: int = Field(ge=1)
    text: str
    char_count: int = Field(ge=0)


class ExtractedDocument(BaseModel):
    """Output of the PDF / DOCX extraction services (Phase 1)."""
    filename: str
    file_type: Literal["pdf", "docx"]
    page_count: int = Field(ge=0)
    text: str
    pages: list[Page]
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_empty: bool


class UploadResponse(BaseModel):
    status: Literal["success"] = "success"
    message: str
    document: ExtractedDocument


# Used in Phase 4: the LLM output is parsed into the right class
# automatically, based on its "document_type" field.
StructuredDocument = Annotated[
    CVDocument | CoverLetterDocument,
    Field(discriminator="document_type"),
]