"""Turn cleaned document text into validated structured data with the LLM."""

from collections.abc import Callable

from app.prompts.document_prompts import CV_SYSTEM_PROMPT, build_cv_user_prompt
from app.schemas.cv import CVDocument
from app.services.llm_service import DEFAULT_MODEL, LLMResult, generate_structured


def extract_cv(
    text: str,
    model: str = DEFAULT_MODEL,
    on_token: Callable[[str], None] | None = None,
) -> LLMResult[CVDocument]:
    """Extract a validated CVDocument from the plain text of a CV.

    Raises:
        ValueError: the text is empty.
        LLMError (and subclasses): see llm_service.
    """
    if not text or not text.strip():
        raise ValueError("Cannot extract a CV from empty text.")

    return generate_structured(
        system_prompt=CV_SYSTEM_PROMPT,
        user_prompt=build_cv_user_prompt(text),
        schema=CVDocument,
        model=model,
        on_token=on_token,
    )