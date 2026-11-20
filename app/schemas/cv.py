"""Structured representation of a CV / resume."""

from typing import Literal

from pydantic import Field, field_validator, model_validator

from app.schemas.common import (
    PRESENT,
    BaseSchema,
    ContactInfo,
    Language,
    PartialDate,
    date_sort_key,
)


def _dedupe(items: list[str]) -> list[str]:
    """Remove empty and duplicate entries (case-insensitive), keep order."""
    seen, result = set(), []
    for item in items:
        item = item.strip()
        if item and item.lower() not in seen:
            seen.add(item.lower())
            result.append(item)
    return result


class DatedEntry(BaseSchema):
    """Anything with a start and end date: a job, a degree..."""
    start_date: PartialDate = None
    end_date: PartialDate = None
    is_current: bool = False

    @model_validator(mode="after")
    def check_dates(self):
        if self.start_date == PRESENT:
            raise ValueError("start_date cannot be 'present'")
        if self.end_date == PRESENT:
            self.is_current = True
            self.end_date = None

        start = date_sort_key(self.start_date)
        end = date_sort_key(self.end_date, is_end=True)
        if start and end and end < start:
            raise ValueError(
                f"end_date ({self.end_date}) is before start_date ({self.start_date})"
            )
        return self


class SkillGroup(BaseSchema):
    category: str | None = None
    skills: list[str] = Field(default_factory=list)

    @field_validator("skills")
    @classmethod
    def dedupe_skills(cls, value: list[str]) -> list[str]:
        return _dedupe(value)


class Experience(DatedEntry):
    title: str
    company: str | None = None
    location: str | None = None
    employment_type: str | None = None   # "Freelance", "Internship", "Full-time"...
    achievements: list[str] = Field(default_factory=list)


class Education(DatedEntry):
    degree: str
    institution: str | None = None
    location: str | None = None


class CVDocument(BaseSchema):
    document_type: Literal["cv"] = "cv"
    contact: ContactInfo = Field(default_factory=ContactInfo)
    headline: str | None = None
    summary: str | None = None
    skills: list[SkillGroup] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    languages: list[Language] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    extracurricular: list[str] = Field(default_factory=list)

    @property
    def all_skills(self) -> list[str]:
        """Every skill from every group, without duplicates."""
        return _dedupe([s for group in self.skills for s in group.skills])