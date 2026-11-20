"""Turn cleaned document text into validated structured data with the LLM."""

from collections.abc import Callable
from typing import Any, Literal

from app.prompts.document_prompts import (
    COVER_LETTER_PROMPT_VERSION,
    COVER_LETTER_SYSTEM_PROMPT,
    CV_PROMPT_VERSION,
    CV_SYSTEM_PROMPT,
    build_cover_letter_user_prompt,
    build_cv_user_prompt,
)
from app.schemas.cover_letter import CoverLetterDocument
from app.schemas.cv import CVDocument
from app.services.classifier import Classification, classify_document
from app.services.llm_service import DEFAULT_MODEL, LLMResult, generate_structured
from app.services.postprocess import fix_cover_letter, fix_cv
from app.services.quality_service import review

RequestedType = Literal["auto", "cv", "cover_letter"]


def _require_text(text: str) -> None:
    if not text or not text.strip():
        raise ValueError("Cannot extract from empty text.")


def extract_cv(
    text: str,
    model: str = DEFAULT_MODEL,
    on_token: Callable[[str], None] | None = None,
) -> LLMResult[CVDocument]:
    """Extract a validated CVDocument from the plain text of a CV."""
    _require_text(text)
    result = generate_structured(
        system_prompt=CV_SYSTEM_PROMPT,
        user_prompt=build_cv_user_prompt(text),
        schema=CVDocument,
        model=model,
        on_token=on_token,
    )
    fix_cv(result.data, text)
    return result


def extract_cover_letter(
    text: str,
    model: str = DEFAULT_MODEL,
    on_token: Callable[[str], None] | None = None,
) -> LLMResult[CoverLetterDocument]:
    """Extract a validated CoverLetterDocument from the plain text of a letter."""
    _require_text(text)
    result = generate_structured(
        system_prompt=COVER_LETTER_SYSTEM_PROMPT,
        user_prompt=build_cover_letter_user_prompt(text),
        schema=CoverLetterDocument,
        model=model,
        on_token=on_token,
    )
    fix_cover_letter(result.data, text)
    return result


_EXTRACTORS = {
    "cv": (extract_cv, CV_PROMPT_VERSION),
    "cover_letter": (extract_cover_letter, COVER_LETTER_PROMPT_VERSION),
}


def process_document(
    text: str,
    requested_type: RequestedType = "auto",
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Full intelligence pipeline: classify (unless the type is given), then extract.

    Returns a plain dict, ready to be stored in a job and sent as JSON.
    """
    _require_text(text)

    if requested_type == "auto":
        classification = classify_document(text, model=model)
    else:
        classification = Classification(requested_type, "user")

    outcome: dict[str, Any] = {
        "document_type": classification.document_type,
        "classification": {
            "method": classification.method,
            "cv_signals": classification.cv_signals,
            "letter_signals": classification.letter_signals,
            "reason": classification.reason,
        },
    }

    if classification.document_type == "other":
        outcome.update(result=None, message="Not a CV or a cover letter: nothing was extracted.")
        return outcome

    extractor, prompt_version = _EXTRACTORS[classification.document_type]
    llm = extractor(text, model=model)
    outcome.update(
        result=llm.data.model_dump(mode="json"),
        review=review(llm.data, text).model_dump(),
        llm={
            "model": llm.model,
            "prompt_version": prompt_version,
            "attempts": llm.attempts,
            "duration_s": llm.duration_s,
            "prompt_tokens": llm.prompt_tokens,
            "output_tokens": llm.output_tokens,
        },
    )
    return outcome