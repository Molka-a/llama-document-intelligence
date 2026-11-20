"""Structured representation of a CV / resume.

Field descriptions are not just documentation: they are sent to the LLM
inside the JSON schema and tell it what belongs in each field.
Field ORDER matters too: the model writes the fields in this order, so it
follows the order of the information in a typical CV (title, company,
dates, then bullet points).
"""

import re
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

_DATE_HELP = "as written in the CV, e.g. 'Mar 2024', '03/2024' or '2024'"
_EMPLOYMENT_IN_TITLE = re.compile(
    r"\s*\((freelance|internship|part[- ]time|full[- ]time|contract|temporary|"
    r"apprenticeship|working student|werkstudent)\)\s*$",
    re.IGNORECASE,
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


_LEADING_BULLET = re.compile(r"^\s*[-•*·]\s+")
_CATEGORY_LINE = re.compile(r"^([^:,()]{2,60}):\s*(.+)$")


def _strip_bullets(items: list[str]) -> list[str]:
    """'- Built a pipeline' -> 'Built a pipeline' (the model copies the bullet marker)."""
    return [_LEADING_BULLET.sub("", item).strip() for item in items if item and item.strip()]


def _split_top_level_commas(text: str) -> list[str]:
    """'Python (Pandas, NumPy), FastAPI' -> ['Python (Pandas, NumPy)', 'FastAPI']."""
    parts, depth, current = [], 0, ""
    for char in text:
        if char in "([":
            depth += 1
        elif char in ")]":
            depth = max(depth - 1, 0)
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    parts.append(current)
    return [p.strip().rstrip(".").strip() for p in parts if p.strip()]


def _check_dates(entry):
    """Shared date rules for jobs and degrees."""
    if entry.start_date == PRESENT:
        raise ValueError("start_date cannot be 'present'")
    if entry.end_date == PRESENT:
        entry.is_current = True
        entry.end_date = None
    elif entry.end_date is not None:
        entry.is_current = False          # a real end date means it is finished

    start = date_sort_key(entry.start_date)
    end = date_sort_key(entry.end_date, is_end=True)
    if start and end and end < start:
        raise ValueError(
            f"end_date ({entry.end_date}) is before start_date ({entry.start_date})"
        )
    return entry


class SkillGroup(BaseSchema):
    category: str | None = Field(
        None, description="Category name as written in the CV, e.g. 'Programming languages'")
    skills: list[str] = Field(
        default_factory=list, description="Individual skills or tools, one per item, e.g. 'Python'")

    @field_validator("skills")
    @classmethod
    def dedupe_skills(cls, value: list[str]) -> list[str]:
        return _dedupe(value)


def _repair_skill_groups(groups: list[SkillGroup]) -> list[SkillGroup]:
    """Split skills the model copied as whole lines.

    'Cloud & tools: Docker, Kubeflow, Azure (basics)' inside a group becomes
    its own group: category 'Cloud & tools', skills ['Docker', 'Kubeflow',
    'Azure (basics)']. Normal skills are left in their group, order is kept.
    """
    repaired = []
    for group in groups:
        pending = []
        for skill in group.skills:
            match = _CATEGORY_LINE.match(skill)
            if match and "," in match.group(2):
                if pending:
                    repaired.append(SkillGroup(category=group.category, skills=pending))
                    pending = []
                repaired.append(SkillGroup(category=match.group(1).strip(),
                                           skills=_split_top_level_commas(match.group(2))))
            else:
                pending.append(skill)
        if pending:
            repaired.append(SkillGroup(category=group.category, skills=pending))
    return repaired


class Experience(BaseSchema):
    title: str = Field(description="Job title only, e.g. 'Data Engineer'")
    company: str | None = Field(None, description="Employer or client name only, e.g. 'Acme GmbH'")
    location: str | None = Field(
        None, description="City or 'Remote', only if written for this job")
    employment_type: str | None = Field(
        None, description="Only if written: 'Freelance', 'Internship', 'Part-time'...")
    start_date: PartialDate = Field(None, description=f"Start date of this job, {_DATE_HELP}")
    end_date: PartialDate = Field(
        None, description=f"End date of this job, {_DATE_HELP}. Use 'Present' if it is ongoing")
    is_current: bool = Field(False, description="True only if the end date is 'Present'")
    achievements: list[str] = Field(
        default_factory=list,
        description="Each bullet point listed under this job, copied as written, one per item")

    @field_validator("achievements")
    @classmethod
    def clean_achievements(cls, value: list[str]) -> list[str]:
        return _strip_bullets(value)

    @model_validator(mode="after")
    def check(self):
        # "Machine Learning Engineer (Freelance)" -> title + employment_type
        match = _EMPLOYMENT_IN_TITLE.search(self.title)
        if match:
            self.title = self.title[:match.start()].strip()
            self.employment_type = self.employment_type or match.group(1).capitalize()
        return _check_dates(self)


class Education(BaseSchema):
    degree: str = Field(description="Degree name, e.g. 'MSc Computer Science'")
    institution: str | None = Field(None, description="School or university name only")
    location: str | None = Field(
        None, description="City and/or country, only if written for this degree")
    start_date: PartialDate = Field(None, description=f"Start date, {_DATE_HELP}")
    end_date: PartialDate = Field(
        None, description=f"End or graduation date, {_DATE_HELP}. Use 'Present' if ongoing")
    is_current: bool = Field(False, description="True only if the end date is 'Present'")
    details: list[str] = Field(
        default_factory=list,
        description="Thesis, grades, honours or other notes listed under this degree, one per item")

    @field_validator("details")
    @classmethod
    def clean_details(cls, value: list[str]) -> list[str]:
        return _strip_bullets(value)

    @model_validator(mode="after")
    def check(self):
        return _check_dates(self)


class CVDocument(BaseSchema):
    document_type: Literal["cv"] = "cv"
    contact: ContactInfo = Field(default_factory=ContactInfo)
    headline: str | None = Field(
        None, description="The professional title line directly under the name, "
                          "e.g. 'Backend Developer | Go, Kubernetes'")
    summary: str | None = Field(
        None, description="The paragraph under a heading like PROFILE, SUMMARY or ABOUT, copied as written")
    skills: list[SkillGroup] = Field(
        default_factory=list, description="Skills grouped by the categories used in the CV")
    experience: list[Experience] = Field(
        default_factory=list, description="Every job, internship or freelance role, in CV order")
    education: list[Education] = Field(
        default_factory=list, description="Every degree, in CV order")
    languages: list[Language] = Field(
        default_factory=list, description="Spoken languages with their level")
    certifications: list[str] = Field(
        default_factory=list, description="Certifications, one per item")
    extracurricular: list[str] = Field(
        default_factory=list, description="Projects, associations, volunteering, one per item")

    @field_validator("certifications", "extracurricular")
    @classmethod
    def clean_lists(cls, value: list[str]) -> list[str]:
        return _strip_bullets(value)

    @model_validator(mode="after")
    def repair_skills(self):
        self.skills = _repair_skill_groups(self.skills)
        return self

    @property
    def all_skills(self) -> list[str]:
        """Every skill from every group, without duplicates."""
        return _dedupe([s for group in self.skills for s in group.skills])