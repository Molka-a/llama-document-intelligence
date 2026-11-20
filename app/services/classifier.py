"""Decide whether a document is a CV, a cover letter or something else.

Two stages:
1. Keyword rules (instant): CV section headings vs. letter phrases.
   They settle most real documents without calling the model.
2. LLM fallback (about 1 minute on CPU), only when the rules are not
   conclusive. It reads just the beginning of the document.
"""

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from app.prompts.document_prompts import CLASSIFY_SYSTEM_PROMPT, build_classify_user_prompt
from app.schemas.common import BaseSchema
from app.services.llm_service import DEFAULT_MODEL, generate_structured

DocType = Literal["cv", "cover_letter", "other"]

# Section headings typical of a CV (EN / FR / DE), matched on short lines only
_CV_HEADING = re.compile(
    r"^(professional |work |relevant |technical |key |core )?"
    r"(experience|experiences|employment|work history|education|skills|profile|summary|"
    r"languages|certifications|projects|"
    r"exp[ée]riences? professionnelles?|formation|comp[ée]tences|langues|"
    r"berufserfahrung|ausbildung|kenntnisse|sprachen)\b",
    re.IGNORECASE,
)

# Phrases typical of a cover letter (EN / FR / DE)
_LETTER_PATTERNS = [
    re.compile(r"^(dear|to whom it may concern|madame|monsieur|sehr geehrte)", re.I | re.M),
    re.compile(r"(kind|best|warm) regards|sincerely|yours faithfully|cordialement|"
               r"mit freundlichen gr(ü|ue)(ß|ss)en", re.I),
    re.compile(r"\bI am (writing|applying)\b|\bje me permets\b|\bhiermit bewerbe\b", re.I),
    re.compile(r"\b(application|candidature|bewerbung)\b", re.I),
]

MIN_SIGNALS = 2          # a type needs at least this many signals to win without the LLM


@dataclass
class Classification:
    document_type: DocType
    method: Literal["rules", "llm", "user"]
    cv_signals: int = 0
    letter_signals: int = 0
    reason: str | None = None


class _LLMClassification(BaseSchema):
    document_type: DocType = Field(description="'cv', 'cover_letter' or 'other'")
    reason: str | None = Field(None, description="One short sentence explaining the choice")


def count_signals(text: str) -> tuple[int, int]:
    """Return (CV signals, cover-letter signals) found in the text."""
    headings = {
        match.group(0).lower()
        for line in text.splitlines()
        if 0 < len(line.strip()) <= 50 and (match := _CV_HEADING.match(line.strip()))
    }
    letter = sum(1 for pattern in _LETTER_PATTERNS if pattern.search(text))
    return len(headings), letter


def classify_document(text: str, model: str = DEFAULT_MODEL) -> Classification:
    cv, letter = count_signals(text)

    if cv >= MIN_SIGNALS and cv > letter:
        return Classification("cv", "rules", cv, letter, f"{cv} CV section headings found")
    if letter >= MIN_SIGNALS and letter > cv:
        return Classification("cover_letter", "rules", cv, letter, f"{letter} letter phrases found")

    result = generate_structured(
        system_prompt=CLASSIFY_SYSTEM_PROMPT,
        user_prompt=build_classify_user_prompt(text),
        schema=_LLMClassification,
        model=model,
    )
    return Classification(result.data.document_type, "llm", cv, letter, result.data.reason)