"""Structured representation of a cover letter.

As for the CV, descriptions and field order are sent to the LLM.
"""

from typing import Literal

from pydantic import Field, field_validator

from app.schemas.common import BaseSchema, ContactInfo


class CoverLetterDocument(BaseSchema):
    document_type: Literal["cover_letter"] = "cover_letter"
    sender: ContactInfo = Field(
        default_factory=ContactInfo,
        description="The candidate who wrote the letter (name and contact details in the header or signature)")
    recipient_name: str | None = Field(
        None, description="Who the greeting addresses: 'Dear Hiring Team,' gives 'Hiring Team', "
                          "'Dear Ms. Schmidt,' gives 'Ms. Schmidt'")
    company: str | None = Field(None, description="The company the candidate is applying to")
    position: str | None = Field(
        None, description="The job title ONLY, without city, dates or cohort: "
                          "'Application: Data Engineer, Munich (2027)' gives 'Data Engineer'")
    job_reference: str | None = Field(None, description="Job ID or reference number, only if written")
    letter_date: str | None = Field(None, description="Date of the letter, as written, e.g. '18 November 2026'")
    key_skills: list[str] = Field(
        default_factory=list,
        description="Skills, tools or technologies the candidate highlights, one per item, e.g. 'Python'")
    motivation_summary: str | None = Field(
        None, description="In 1-2 sentences: why the candidate wants this company or role, "
                          "using the letter's own arguments")
    availability: str | None = Field(
        None, description="Start date, notice period or relocation/visa information, only if written")

    @field_validator("key_skills")
    @classmethod
    def dedupe_skills(cls, value: list[str]) -> list[str]:
        seen, result = set(), []
        for item in value:
            item = item.strip()
            if item and item.lower() not in seen:
                seen.add(item.lower())
                result.append(item)
        return result