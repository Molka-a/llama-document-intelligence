"""Structured representation of a cover letter."""

from typing import Literal

from pydantic import Field

from app.schemas.common import BaseSchema, ContactInfo


class CoverLetterDocument(BaseSchema):
    document_type: Literal["cover_letter"] = "cover_letter"
    sender: ContactInfo = Field(default_factory=ContactInfo)
    recipient_name: str | None = None      # "Ms. Schmidt", "Hiring Manager"
    company: str | None = None
    position: str | None = None
    job_reference: str | None = None       # job ID if mentioned
    letter_date: str | None = None         # kept as written: "18 November 2026"
    key_skills: list[str] = Field(default_factory=list)
    motivation_summary: str | None = None  # 1-3 sentences: why this company/role
    availability: str | None = None        # start date / notice period if mentioned