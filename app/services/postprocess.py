"""Deterministic corrections applied after the LLM, using the source text.

Each rule fixes an error measured in the evaluation (Phase 5), where plain
code is more reliable than the model:

- recipient_name: the 3B model leaves it empty, although the greeting line
  ("Dear Ms. Keller,") states it. It is read from the greeting instead.
- education location: the model invents a city from the school name
  ("University of Hamburg" -> "Hamburg"). A city is kept only if it is
  written on the degree's line, outside the school name.
"""

import re

from app.schemas.cover_letter import CoverLetterDocument
from app.schemas.cv import CVDocument

_GREETING = re.compile(
    r"^\s*(?:dear|hello|hi|sehr geehrte[rs]?|bonjour)\s+(.{2,60}?)\s*[,:!]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_NOT_A_RECIPIENT = ("whom", "sir or madam", "sir/madam", "madam or sir")


def recipient_from_greeting(text: str) -> str | None:
    """'Dear Ms. Keller,' -> 'Ms. Keller'. None for generic greetings."""
    match = _GREETING.search(text)
    if not match:
        return None
    name = match.group(1).strip()
    return None if name.lower().startswith(_NOT_A_RECIPIENT) else name


def fix_cover_letter(letter: CoverLetterDocument, text: str) -> CoverLetterDocument:
    if not letter.recipient_name:
        letter.recipient_name = recipient_from_greeting(text)
    return letter


def _contains(haystack: str, needle: str) -> bool:
    return needle.lower() in haystack.lower()


def _location_written_for(text: str, institution: str | None, degree: str, location: str) -> bool:
    """Is `location` written on the degree's line (or the next one), outside the school name?"""
    lines = text.splitlines()
    anchors = [a for a in (institution, degree) if a]
    for i, line in enumerate(lines):
        if not any(_contains(line, a) for a in anchors):
            continue
        nearby = " ".join(lines[i:i + 2])
        if institution:
            nearby = re.sub(re.escape(institution), " ", nearby, flags=re.IGNORECASE)
        if re.search(rf"\b{re.escape(location)}\b", nearby, re.IGNORECASE):
            return True
    return False


def fix_cv(cv: CVDocument, text: str) -> CVDocument:
    for degree in cv.education:
        if degree.location and not _location_written_for(
                text, degree.institution, degree.degree, degree.location):
            degree.location = None
    return cv