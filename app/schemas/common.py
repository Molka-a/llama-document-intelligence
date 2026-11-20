"""Shared building blocks for all document schemas."""

import re
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator


class BaseSchema(BaseModel):
    """Base for every schema.

    - Strips spaces around all strings.
    - Ignores unknown keys (an LLM often adds extra fields).
    """
    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

_MONTHS = {
    "jan": 1, "january": 1, "janvier": 1, "januar": 1,
    "feb": 2, "february": 2, "fev": 2, "février": 2, "fevrier": 2, "februar": 2,
    "mar": 3, "march": 3, "mars": 3, "märz": 3, "maerz": 3,
    "apr": 4, "april": 4, "avr": 4, "avril": 4,
    "may": 5, "mai": 5,
    "jun": 6, "june": 6, "juin": 6, "juni": 6,
    "jul": 7, "july": 7, "juil": 7, "juillet": 7, "juli": 7,
    "aug": 8, "august": 8, "aout": 8, "août": 8,
    "sep": 9, "sept": 9, "september": 9, "septembre": 9,
    "oct": 10, "october": 10, "octobre": 10, "okt": 10, "oktober": 10,
    "nov": 11, "november": 11, "novembre": 11,
    "dec": 12, "december": 12, "décembre": 12, "decembre": 12, "dez": 12, "dezember": 12,
}

_PRESENT = {"present", "current", "now", "today", "ongoing",
            "aujourd'hui", "actuel", "heute", "aktuell"}

PRESENT = "present"


def normalize_date(value) -> str | None:
    """Convert CV-style dates to a standard form.

    Output is one of: "YYYY-MM", "YYYY", "present" or None.

        "Aug 2026"   -> "2026-08"
        "08/2026"    -> "2026-08"
        "2026-08-15" -> "2026-08"
        "2025"       -> "2025"
        "Present"    -> "present"
    """
    if value is None:
        return None
    text = str(value).strip().lower().rstrip(".")
    if not text or text in {"null", "none", "n/a", "-"}:
        return None
    if text in _PRESENT:
        return PRESENT

    if m := re.fullmatch(r"(\d{4})", text):
        return m.group(1)
    if m := re.fullmatch(r"(\d{4})-(\d{1,2})(?:-\d{1,2})?", text):
        year, month = int(m.group(1)), int(m.group(2))
    elif m := re.fullmatch(r"(\d{1,2})[/.](\d{4})", text):
        month, year = int(m.group(1)), int(m.group(2))
    elif m := re.fullmatch(r"([a-zäéû]+)\.?\s+(\d{4})", text):
        month = _MONTHS.get(m.group(1))
        year = int(m.group(2))
        if month is None:
            raise ValueError(f"Unknown month name in date: {value!r}")
    else:
        raise ValueError(f"Unrecognized date format: {value!r}")

    if not 1 <= month <= 12:
        raise ValueError(f"Invalid month in date: {value!r}")
    return f"{year:04d}-{month:02d}"


def date_sort_key(value: str | None, *, is_end: bool = False) -> tuple[int, int] | None:
    """Turn a normalized date into (year, month) for comparison.

    A year alone counts as January for a start date and December for
    an end date, so "2025 - 2025" is a valid range.
    """
    if value is None or value == PRESENT:
        return None
    if len(value) == 4:
        return int(value), 12 if is_end else 1
    year, month = value.split("-")
    return int(year), int(month)


PartialDate = Annotated[str | None, BeforeValidator(normalize_date)]


# ---------------------------------------------------------------------------
# Shared models
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$")
_PHONE_RE = re.compile(r"^\+?[\d\s().-]+$")


class ContactInfo(BaseSchema):
    full_name: str | None = Field(
        None, description="The person's full name, usually the first line of a CV, e.g. 'Jane Doe'")
    email: str | None = Field(None, description="Email address")
    phone: str | None = Field(None, description="Phone number, as written")
    location: str | None = Field(
        None, description="Where the person lives: city and country, e.g. 'Berlin, Germany'")
    linkedin: str | None = Field(None, description="LinkedIn profile URL")
    github: str | None = Field(None, description="GitHub profile URL")
    website: str | None = Field(None, description="Personal website or portfolio URL")
    
    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str | None) -> str | None:
        if not value:
            return None
        value = value.lower()
        if not _EMAIL_RE.match(value):
            raise ValueError(f"Invalid email address: {value!r}")
        return value

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, value: str | None) -> str | None:
        if not value:
            return None
        digits = re.sub(r"\D", "", value)
        if not _PHONE_RE.match(value) or not 6 <= len(digits) <= 15:
            raise ValueError(f"Invalid phone number: {value!r}")
        return re.sub(r"\s+", " ", value)

    @field_validator("linkedin", "github", "website")
    @classmethod
    def strip_url_prefix(cls, value: str | None) -> str | None:
        if not value:
            return None
        return re.sub(r"^(https?://)?(www\.)?", "", value).rstrip("/")


class Language(BaseSchema):
    name: str
    level: str | None = None   # free text as written: "native", "fluent", "B2"...