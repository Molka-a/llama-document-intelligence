"""Quality review of an extraction: missing fields and uncertain (ungrounded) values.

- Missing: important fields left empty by the model.
- Uncertain: values that cannot be found in the source text. A small model
  can invent a city, a company or a skill; a value that appears nowhere in
  the document is flagged so a person can check it.

The check is deterministic and fast (no LLM). It is deliberately simple:
it catches invented values, not values copied from the wrong place in the
document (e.g. a city that exists elsewhere in the CV).
"""

import re
from typing import Any

from pydantic import BaseModel

from app.schemas.cover_letter import CoverLetterDocument
from app.schemas.cv import CVDocument

# Fields that matter for a usable result (dotted paths, [] = every item)
_CV_KEY_FIELDS = [
    "contact.full_name", "contact.email", "contact.phone", "contact.location",
    "headline", "summary", "skills", "experience", "education", "languages",
    "experience[].company", "experience[].start_date", "education[].institution",
]
_LETTER_KEY_FIELDS = [
    "sender.full_name", "company", "position", "key_skills", "motivation_summary",
]

# Fields not checked against the text: generated or derived by design
_NOT_GROUNDED = {"document_type", "is_current", "motivation_summary", "level"}

MIN_COMPLETENESS = 0.7       # below this, a person should review the result
_SNIPPET = 40                # long values: only the beginning is searched


class UncertainField(BaseModel):
    field: str
    value: str
    reason: str


class QualityReport(BaseModel):
    completeness: float               # share of key fields that are filled (0 to 1)
    missing_fields: list[str]
    uncertain_fields: list[UncertainField]
    needs_review: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def review(document: CVDocument | CoverLetterDocument, source_text: str) -> QualityReport:
    key_fields = _CV_KEY_FIELDS if isinstance(document, CVDocument) else _LETTER_KEY_FIELDS
    data = document.model_dump(mode="json")

    missing = [path for field in key_fields for path in _missing_paths(data, field)]
    checked = sum(len(_expand(data, field)) or 1 for field in key_fields)
    completeness = round(1 - len(missing) / checked, 2) if checked else 1.0

    haystack = _normalize(source_text)
    digits = re.sub(r"\D", "", source_text)
    uncertain = [
        UncertainField(field=path, value=value, reason=reason)
        for path, value in _string_leaves(data)
        if (reason := _grounding_problem(path, value, haystack, digits))
    ]

    return QualityReport(
        completeness=completeness,
        missing_fields=missing,
        uncertain_fields=uncertain,
        needs_review=bool(uncertain) or completeness < MIN_COMPLETENESS,
    )


# ---------------------------------------------------------------------------
# Missing fields
# ---------------------------------------------------------------------------

def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _expand(data: dict, field: str) -> list[tuple[str, Any]]:
    """'experience[].company' -> [('experience[0].company', value), ...]."""
    results = [("", data)]
    for part in field.split("."):
        is_list = part.endswith("[]")
        name = part[:-2] if is_list else part
        next_results = []
        for path, node in results:
            value = node.get(name) if isinstance(node, dict) else None
            base = f"{path}.{name}" if path else name
            if is_list:
                next_results += [(f"{base}[{i}]", item) for i, item in enumerate(value or [])]
            else:
                next_results.append((base, value))
        results = next_results
    return results


def _missing_paths(data: dict, field: str) -> list[str]:
    return [path for path, value in _expand(data, field) if _is_empty(value)]


# ---------------------------------------------------------------------------
# Grounding (is the value really in the document?)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://(www\.)?", "", text)
    text = re.sub(r"[^\w@./+%]+", " ", text)      # punctuation and line breaks -> space
    return re.sub(r"\s+", " ", text).strip()


def _string_leaves(node: Any, path: str = ""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _string_leaves(value, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _string_leaves(value, f"{path}[{i}]")
    elif isinstance(node, str) and node.strip():
        yield path, node


def _grounding_problem(path: str, value: str, haystack: str, digits: str) -> str | None:
    field = re.sub(r"\[\d+\]", "", path).split(".")[-1]
    if field in _NOT_GROUNDED:
        return None

    if field == "phone":
        found = re.sub(r"\D", "", value) in digits
    elif field in ("start_date", "end_date"):
        found = value[:4] in digits                   # the year must appear in the document
    else:
        needle = _normalize(value)[:_SNIPPET].strip()
        found = not needle or needle in haystack

    return None if found else "not found in the document text"