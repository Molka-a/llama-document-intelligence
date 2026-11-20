"""Extraction service tests (no Ollama needed: the LLM call is replaced)."""

import pytest

from app.prompts.document_prompts import CV_SYSTEM_PROMPT, build_cv_user_prompt
from app.schemas.cv import CVDocument
from app.services import extraction_service
from app.services.llm_service import LLMResult


def test_extract_cv_sends_prompt_text_and_schema(monkeypatch):
    calls = {}

    def fake_generate(system_prompt, user_prompt, schema, model, on_token=None):
        calls.update(system=system_prompt, user=user_prompt, schema=schema, model=model)
        data = schema.model_validate({"contact": {"full_name": "Jane Doe"}})
        return LLMResult(data=data, model=model, attempts=1, duration_s=1.0,
                         prompt_tokens=10, output_tokens=5)

    monkeypatch.setattr(extraction_service, "generate_structured", fake_generate)

    result = extraction_service.extract_cv("JANE DOE\nData Engineer", model="test-model")

    assert result.data.contact.full_name == "Jane Doe"
    assert calls["system"] == CV_SYSTEM_PROMPT
    assert "JANE DOE" in calls["user"]
    assert calls["schema"] is CVDocument
    assert calls["model"] == "test-model"


@pytest.mark.parametrize("text", ["", "   \n  "])
def test_extract_cv_rejects_empty_text(text):
    with pytest.raises(ValueError, match="empty"):
        extraction_service.extract_cv(text)


def test_user_prompt_wraps_text_in_delimiters():
    prompt = build_cv_user_prompt("  JANE DOE  ")
    assert prompt == "CV text:\n<<<\nJANE DOE\n>>>"


def test_cv_schema_descriptions_reach_the_llm():
    """Descriptions are what helped the model find the name and location."""
    schema = str(CVDocument.model_json_schema())
    assert "Job title only" in schema
    assert "Each bullet point listed under this job" in schema


def test_education_details_field():
    cv = CVDocument.model_validate({
        "education": [{"degree": "MSc Data Science", "details": ["Thesis: 17/20"]}]
    })
    assert cv.education[0].details == ["Thesis: 17/20"]