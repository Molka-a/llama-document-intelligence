"""Cover letter extraction and the full process_document pipeline (no Ollama needed)."""

import pytest

from app.prompts.document_prompts import COVER_LETTER_SYSTEM_PROMPT
from app.schemas.cover_letter import CoverLetterDocument
from app.services import extraction_service
from app.services.classifier import Classification
from app.services.llm_service import LLMResult


def fake_extractor(data: dict):
    def _extract(text, model="m"):
        return LLMResult(data=data["schema"].model_validate(data["values"]), model=model,
                         attempts=1, duration_s=2.5, prompt_tokens=100, output_tokens=50)
    return _extract


def test_extract_cover_letter_uses_letter_prompt_and_schema(monkeypatch):
    calls = {}

    def fake_generate(system_prompt, user_prompt, schema, model, on_token=None):
        calls.update(system=system_prompt, user=user_prompt, schema=schema)
        return LLMResult(data=schema(company="Acme"), model=model, attempts=1,
                         duration_s=1.0, prompt_tokens=1, output_tokens=1)

    monkeypatch.setattr(extraction_service, "generate_structured", fake_generate)

    result = extraction_service.extract_cover_letter("Dear team, ...")

    assert result.data.company == "Acme"
    assert calls["system"] == COVER_LETTER_SYSTEM_PROMPT
    assert calls["schema"] is CoverLetterDocument
    assert "Dear team" in calls["user"]


def test_cover_letter_skills_are_deduplicated():
    letter = CoverLetterDocument.model_validate({"key_skills": ["Python", "python", " LLMs "]})
    assert letter.key_skills == ["Python", "LLMs"]


def test_process_document_auto_cv(monkeypatch):
    monkeypatch.setattr(extraction_service, "classify_document",
                        lambda text, model: Classification("cv", "rules", 5, 0, "5 headings"))
    monkeypatch.setattr(extraction_service, "_EXTRACTORS", {
        "cv": (fake_extractor({"schema": extraction_service.CVDocument,
                               "values": {"contact": {"full_name": "Jane Doe"}}}), "cv-test"),
    })

    outcome = extraction_service.process_document("JANE DOE ...")

    assert outcome["document_type"] == "cv"
    assert outcome["classification"]["method"] == "rules"
    assert outcome["result"]["contact"]["full_name"] == "Jane Doe"
    assert outcome["llm"]["prompt_version"] == "cv-test"
    assert outcome["llm"]["duration_s"] == 2.5
    assert outcome["review"]["needs_review"] is True       # most CV fields are empty
    assert "contact.email" in outcome["review"]["missing_fields"]


def test_process_document_user_type_skips_classification(monkeypatch):
    def must_not_classify(*args, **kwargs):
        raise AssertionError("classification should be skipped")

    monkeypatch.setattr(extraction_service, "classify_document", must_not_classify)
    monkeypatch.setattr(extraction_service, "_EXTRACTORS", {
        "cover_letter": (fake_extractor({"schema": CoverLetterDocument,
                                         "values": {"company": "Acme"}}), "cl-test"),
    })

    outcome = extraction_service.process_document("Dear team", requested_type="cover_letter")

    assert outcome["classification"]["method"] == "user"
    assert outcome["result"]["company"] == "Acme"


def test_process_document_other_extracts_nothing(monkeypatch):
    monkeypatch.setattr(extraction_service, "classify_document",
                        lambda text, model: Classification("other", "llm", 0, 0, "An invoice"))

    outcome = extraction_service.process_document("INVOICE 2026-114")

    assert outcome["document_type"] == "other"
    assert outcome["result"] is None
    assert "nothing was extracted" in outcome["message"]


def test_process_document_rejects_empty_text():
    with pytest.raises(ValueError, match="empty"):
        extraction_service.process_document("   ")
